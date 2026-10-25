"""End-to-end: run the CLI against the fake pipeline, break it, and check the gate catches it."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ragcheck.cli import app
from ragcheck.config import load_config, load_golden
from ragcheck.judges import CachedJudge
from ragcheck.runner import run
from ragcheck.targets import make_target

cli = CliRunner()


def invoke(*args: str) -> Any:
    return cli.invoke(app, list(args), catch_exceptions=False)


def test_good_run_passes_and_becomes_baseline(project: Path) -> None:
    r = invoke("run", "--save-baseline", "--md", "summary.md")
    assert r.exit_code == 0, r.output
    assert (project / ".ragcheck" / "baseline.json").exists()
    assert (project / "ragcheck-report.html").read_text(encoding="utf-8").count('class="case"') == 4
    assert "RAGCheck passed" in (project / "summary.md").read_text(encoding="utf-8")


def test_broken_retriever_fails_the_gate(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert invoke("run", "--save-baseline").exit_code == 0
    monkeypatch.setenv("RAGCHECK_TEST_MODE", "broken")
    r = invoke("run", "--save-baseline", "--md", "summary.md")
    assert r.exit_code == 1
    assert "hit_rate dropped" in r.output
    assert "not saved as baseline" in r.output
    md = (project / "summary.md").read_text(encoding="utf-8")
    assert "question(s) regressed" in md and "`refund`" in md


def test_unsafe_pipeline_fails_guardrails(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGCHECK_TEST_MODE", "unsafe")
    r = invoke("run")
    assert r.exit_code == 1
    for rail in ("pii", "injection", "refusal"):
        assert f"Guardrail '{rail}' failed" in r.output


def test_compare_rechecks_saved_runs(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    invoke("run", "--save-baseline")
    good = sorted((project / ".ragcheck" / "runs").glob("*.json"))[-1]
    monkeypatch.setenv("RAGCHECK_TEST_MODE", "broken")
    invoke("run")
    assert invoke("compare").exit_code == 1  # latest (broken) run versus the good baseline
    assert invoke("compare", str(good)).exit_code == 0


def test_bad_config_exits_2(project: Path) -> None:
    (project / "ragcheck.yaml").write_text("target: {type: carrier-pigeon}", encoding="utf-8")
    assert invoke("run").exit_code == 2


def test_init_writes_starter_files(tmp_path: Path) -> None:
    r = invoke("init", str(tmp_path))
    assert r.exit_code == 0
    assert load_config(tmp_path / "ragcheck.yaml").target.type == "http"
    assert len(load_golden(tmp_path / "golden.jsonl")) == 4


class ScriptedJudge:
    """Answers the judge prompts deterministically, counting calls."""

    name = "scripted"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, system: str, prompt: str) -> str:
        self.calls += 1
        if "Split the ANSWER" in prompt:
            return json.dumps(
                {"claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}]}
            )
        if "decline" in prompt:
            return json.dumps({"declined": True})
        return json.dumps({"score": 5, "reason": "fine"})

    async def aclose(self) -> None:
        return None


def test_judged_metrics_and_cache(project: Path) -> None:
    cfg = load_config(project / "ragcheck.yaml")
    cases = load_golden(project / "golden.jsonl")
    inner = ScriptedJudge()

    def once() -> Any:
        judge = CachedJudge(inner, project / ".ragcheck" / "cache")
        return asyncio.run(run(cfg, cases, make_target(cfg), judge)), judge

    first, j1 = once()
    assert first.metrics["faithfulness"] == 0.5
    assert first.metrics["relevance"] == 1.0
    assert first.metrics["correctness"] == 1.0  # only the case with a reference answer
    assert "correctness" not in next(r for r in first.results if r.case.id == "shipping").scores
    calls = inner.calls
    second, j2 = once()
    assert inner.calls == calls and j2.hits == j1.misses  # every judge call served from the cache
    assert second.metrics == first.metrics
