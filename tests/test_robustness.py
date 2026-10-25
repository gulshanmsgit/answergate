"""Adversarial cases: hostile pipelines, flaky judges, bad files and unfair comparisons.

Each test describes a way a real user's setup can misbehave; RAGCheck must either handle it
or fail with a clear message, never crash the whole run or give a wrong verdict.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from ragcheck.baseline import evaluate
from ragcheck.cli import app
from ragcheck.config import ConfigError, Thresholds, load_config, load_golden
from ragcheck.guardrails import find_pii
from ragcheck.judges import CachedJudge
from ragcheck.models import CaseResult, GoldenCase, RunResult, TargetResponse
from ragcheck.runner import aggregate, run

cli = CliRunner()


class FnTarget:
    def __init__(self, fn: Any) -> None:
        self.fn = fn

    async def ask(self, question: str) -> TargetResponse:
        out = self.fn(question)
        if asyncio.iscoroutine(out):
            out = await out
        from ragcheck.targets.parse import to_response

        return to_response(out)

    async def aclose(self) -> None:
        return None


class FnJudge:
    name = "fn"

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self.calls = 0

    async def complete(self, system: str, prompt: str) -> str:
        self.calls += 1
        return str(self.fn(prompt, self.calls))

    async def aclose(self) -> None:
        return None


def go(project: Path, target: Any, judge_fn: Any = None, **cfg_overrides: Any) -> RunResult:
    cfg = load_config(project / "ragcheck.yaml")
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    judge = CachedJudge(FnJudge(judge_fn), project / ".ragcheck" / "cache") if judge_fn else None
    return asyncio.run(run(cfg, load_golden(project / "golden.jsonl"), target, judge))


GOOD = {"answer": "Refunds within 30 days.", "contexts": [{"id": "refund-policy.md#0", "text": "30 days"}]}


def ok_judge(prompt: str, n: int) -> str:
    if "Split the ANSWER" in prompt:
        return json.dumps({"claims": [{"claim": "x", "supported": True}]})
    if "declined" in prompt:
        return json.dumps({"declined": True})
    return json.dumps({"score": 5, "reason": "ok"})


# ── Pipeline misbehaviour ─────────────────────────────────────────────


def test_pipeline_that_hangs_times_out(project: Path) -> None:
    async def slow(q: str) -> dict[str, Any]:
        await asyncio.sleep(30)
        return GOOD

    r = go(project, FnTarget(slow), target_timeout_s=0.2, retries=0)
    assert r.errors == 4 and all("Timeout" in (c.error or "") for c in r.results)


def test_pipeline_returning_garbage_is_a_case_error_not_a_crash(project: Path) -> None:
    outs = iter([None, 42, {"answer": None}, {"answer": "x", "contexts": "not-a-list"}])
    r = go(project, FnTarget(lambda q: next(outs)), retries=0)
    assert r.errors == 4


def test_null_contexts_means_no_contexts(project: Path) -> None:
    r = go(project, FnTarget(lambda q: {"answer": "hi", "contexts": None}), retries=0)
    assert r.errors == 0


def test_one_flaky_call_is_retried(project: Path) -> None:
    calls: dict[str, int] = {}

    def flaky(q: str) -> dict[str, Any]:
        calls[q] = calls.get(q, 0) + 1
        if calls[q] == 1:
            raise ConnectionError("reset by peer")
        return GOOD

    r = go(project, FnTarget(flaky), retries=1)
    assert r.errors == 0


# ── Judge misbehaviour ────────────────────────────────────────────────


def test_judge_http_error_does_not_crash_the_run(project: Path) -> None:
    def boom(prompt: str, n: int) -> str:
        req = httpx.Request("POST", "http://judge")
        raise httpx.HTTPStatusError("500", request=req, response=httpx.Response(500, request=req))

    r = go(project, FnTarget(lambda q: GOOD), boom, retries=0)
    assert len(r.results) == 4  # every case still reported
    assert r.judge_errors == 4 and r.errors == 0  # the pipeline itself was fine


def test_judge_timeout_on_refusal_check_does_not_crash(project: Path) -> None:
    def slow(prompt: str, n: int) -> str:
        if "declined" in prompt:
            raise httpx.ReadTimeout("judge too slow")
        return ok_judge(prompt, n)

    r = go(project, FnTarget(lambda q: GOOD), slow, retries=0)
    weather = next(c for c in r.results if c.case.id == "weather")
    # Falls back to the heuristic rather than dying
    assert "refusal" in weather.guardrails


def test_judge_bad_json_once_is_retried(project: Path) -> None:
    def sometimes(prompt: str, n: int) -> str:
        return "Sure! Here is my rating:" if n == 1 else ok_judge(prompt, n)

    r = go(project, FnTarget(lambda q: GOOD), sometimes, retries=0)
    assert r.judge_errors == 0


@pytest.mark.parametrize("score", ["4", "4/5", "four", None, 9, -3, [4]])
def test_judge_odd_score_values(project: Path, score: Any) -> None:
    def odd(prompt: str, n: int) -> str:
        if "Split the ANSWER" in prompt:
            return json.dumps(
                {"claims": [{"claim": "x", "supported": "true"}, {"claim": "y", "supported": "no"}]}
            )
        return json.dumps({"score": score, "declined": True})

    r = go(project, FnTarget(lambda q: GOOD), odd, retries=0)
    refund = next(c for c in r.results if c.case.id == "refund")
    assert refund.scores["faithfulness"] == 0.5  # "true" string counts as supported
    rel = refund.scores.get("relevance")
    assert rel is None or 0.0 <= rel <= 1.0


def test_corrupted_cache_file_is_ignored(project: Path) -> None:
    go(project, FnTarget(lambda q: GOOD), ok_judge)
    for f in (project / ".ragcheck" / "cache").rglob("*.json"):
        f.write_text("{trunc", encoding="utf-8")
    r = go(project, FnTarget(lambda q: GOOD), ok_judge)
    assert r.judge_errors == 0


# ── Bad input files ───────────────────────────────────────────────────


def test_golden_set_saved_with_bom_by_notepad(project: Path) -> None:
    p = project / "golden.jsonl"
    p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())
    assert len(load_golden(p)) == 4


def test_golden_set_as_json_array(project: Path) -> None:
    p = project / "golden.jsonl"
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    assert len(load_golden(p)) == 4


def test_invalid_k_is_rejected(project: Path) -> None:
    (project / "ragcheck.yaml").write_text("target: {type: python, callable: fake_rag:answer}\nk: 0\n")
    with pytest.raises(ConfigError):
        load_config(project / "ragcheck.yaml")


def test_corrupt_baseline_gives_clean_error(project: Path) -> None:
    (project / ".ragcheck").mkdir()
    (project / ".ragcheck" / "baseline.json").write_text("{}", encoding="utf-8")
    r = cli.invoke(app, ["run"])
    assert r.exit_code == 2 and "baseline" in r.output.lower()


def test_unexpected_crash_exits_2_not_1(project: Path) -> None:
    # Exit code 1 means "quality dropped"; a RAGCheck bug must never look like that in CI
    # A fresh module name: fake_rag is already imported by earlier tests in this process
    (project / "broken_mod.py").write_text("raise SystemError('import-time bug')\n", encoding="utf-8")
    cfg = (project / "ragcheck.yaml").read_text(encoding="utf-8").replace("fake_rag:", "broken_mod:")
    (project / "ragcheck.yaml").write_text(cfg, encoding="utf-8")
    r = cli.invoke(app, ["run"])
    assert r.exit_code == 2


# ── Gate fairness ─────────────────────────────────────────────────────


def mk(rid: str, rows: dict[str, float]) -> RunResult:
    results = [CaseResult(case=GoldenCase(id=i, question=i), scores={"hit_rate": s}) for i, s in rows.items()]
    m, g = aggregate(results)
    return RunResult(id=rid, created_at="", results=results, metrics=m, guardrails=g)


def test_adding_hard_questions_is_not_a_regression() -> None:
    base = mk("b", {"a": 1.0, "b": 1.0})
    cur = mk("c", {"a": 1.0, "b": 1.0, "new1": 0.0, "new2": 0.0})  # same answers + 2 new hard ones
    v = evaluate(cur, base, Thresholds(default_max_drop=0.05))
    assert v.passed, v.failures
    assert any("2 new" in w for w in v.warnings)


def test_removing_failing_questions_does_not_hide_a_regression() -> None:
    base = mk("b", {"a": 1.0, "b": 1.0, "c": 0.0})
    cur = mk("c", {"a": 0.0, "b": 1.0})  # "a" broke; "c" was deleted from the golden set
    v = evaluate(cur, base, Thresholds(default_max_drop=0.05))
    assert not v.passed


def test_judge_skipped_this_run_is_flagged() -> None:
    base = RunResult(id="b", created_at="", results=[], metrics={"faithfulness": 0.9})
    cur = RunResult(id="c", created_at="", results=[], metrics={})
    v = evaluate(cur, base, Thresholds())
    assert not v.passed  # a metric silently disappearing must not pass the gate


def test_save_baseline_refuses_a_failing_first_run(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGCHECK_TEST_MODE", "unsafe")
    cli.invoke(app, ["run", "--save-baseline"])
    assert not (project / ".ragcheck" / "baseline.json").exists()


# ── Report and guardrails ─────────────────────────────────────────────


def test_report_escapes_html_in_answers(project: Path) -> None:
    from ragcheck.report import html

    evil = "<script>alert(1)</script>"
    r = go(project, FnTarget(lambda q: {"answer": evil, "contexts": [{"id": evil, "text": evil}]}))
    page = html(r, evaluate(r, None, Thresholds()), None)
    assert evil not in page and "&lt;script&gt;" in page


@pytest.mark.parametrize(
    "text",
    [
        "Released on 2025-10-25 at 16:05.",
        "Version 1.24.3 supports 128000 tokens.",
        "Our revenue was 1,250,000 dollars in 2024.",
        "See section 4.2.1 and table 12.",
        "Order #1234 ships in 7-14 days.",
    ],
)
def test_pii_no_false_positives_on_ordinary_numbers(text: str) -> None:
    assert find_pii(text) == []


def test_internal_bug_exits_2(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ragcheck.cli as cli_mod

    def bug(*a: Any, **k: Any) -> Any:
        raise RuntimeError("internal bug")

    monkeypatch.setattr(cli_mod, "save_run", bug)
    r = cli.invoke(app, ["run"])
    assert r.exit_code == 2 and "RAGCheck crashed" in r.output
