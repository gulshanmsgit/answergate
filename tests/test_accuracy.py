"""Accuracy safeguards: evidence checking, multi-sample judging, calibration and confidence intervals."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import quote_from_prompt
from typer.testing import CliRunner

from ragcheck.baseline import confidence_interval, evaluate
from ragcheck.calibrate import CUTOFFS, calibrate, cohen_kappa, load_labels
from ragcheck.cli import app
from ragcheck.config import Config, Thresholds
from ragcheck.judges import CachedJudge
from ragcheck.metrics import judged
from ragcheck.metrics.judged import evidence_found
from ragcheck.models import CaseResult, Context, GoldenCase, RunResult
from ragcheck.runner import aggregate

CTX = [
    Context(
        id="refund-policy.md",
        text="Customers can return items within 30 days of delivery for a full refund.\n"
        "Refunds are issued to the original payment method.",
    )
]


class FnJudge:
    name = "fn"

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self.samples_seen: list[int] = []

    async def complete(self, system: str, prompt: str, schema: Any = None, sample: int = 0) -> str:
        self.samples_seen.append(sample)
        return json.dumps(self.fn(prompt, sample))

    async def aclose(self) -> None:
        return None


def judge(tmp: Path, fn: Any) -> CachedJudge:
    return CachedJudge(FnJudge(fn), tmp / "cache", backoff_s=0.01)


# ── Evidence checking ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "quote",
    [
        "Customers can return items within 30 days of delivery",
        "customers can return items within 30 days of delivery for a full refund",  # case
        "“within 30 days of delivery”",  # curly quotes
        "Customers can return items within   30 days\nof delivery",  # whitespace
        "Customers can return items ... issued to the original payment method",  # elided
        "Customers may return items within 30 days of delivery for a full refund",  # one word differs
    ],
)
def test_real_quotes_are_found(quote: str) -> None:
    assert evidence_found(quote, CTX)


def test_quote_with_passage_labels_across_chunks_is_found() -> None:
    # A real judge reply: it copied the passage label and joined two chunks, skipping a markup line
    ctx = [
        Context(
            id="jwt.md#28", text="Create a random secret key.\n\n<div>\n\n```console\n$ openssl rand -hex 32"
        ),
        Context(
            id="jwt.md#29", text="```console\n$ openssl rand -hex 32\n\n09d25e09\n```\n\nAnd copy the output."
        ),
    ]
    quote = (
        "[1] (id: jwt.md#28) Create a random secret key.\n```console\n$ openssl rand -hex 32\n09d25e09\n```"
    )
    assert evidence_found(quote, ctx)
    assert not evidence_found("[1] (id: jwt.md#28) Create a random secret key.\n$ openssl rand -hex 64", ctx)


@pytest.mark.parametrize(
    "quote",
    [
        "",
        "ok",
        "Customers can return items within 60 days of delivery",  # changed number, short
        "Customers can return items within 60 days of delivery for a full refund",  # changed number, long
        "Items can be returned within 90 days for store credit only",
        "Refunds are issued as store credit",
    ],
)
def test_invented_quotes_are_rejected(quote: str) -> None:
    assert not evidence_found(quote, CTX)


def test_judge_saying_supported_with_fake_evidence_is_overruled(tmp_path: Path) -> None:
    def lying(prompt: str, sample: int) -> dict[str, Any]:
        return {
            "claims": [
                {"claim": "30 days", "supported": True, "evidence": "within 30 days of delivery"},
                {"claim": "free shipping", "supported": True, "evidence": "Shipping is always free"},
            ]
        }

    async def main() -> judged.Judged:
        return await judged.faithfulness(judge(tmp_path, lying), "30 days, free shipping", CTX)

    r = asyncio.run(main())
    assert r.score == 0.5 and "overruled" in r.note


def test_evidence_check_can_be_switched_off(tmp_path: Path) -> None:
    def no_evidence(prompt: str, sample: int) -> dict[str, Any]:
        return {"claims": [{"claim": "x", "supported": True, "evidence": ""}]}

    r = asyncio.run(judged.faithfulness(judge(tmp_path, no_evidence), "x", CTX, verify=False))
    assert r.score == 1.0


# ── Multi-sample judging ─────────────────────────────────────────────


def test_median_of_samples_and_spread(tmp_path: Path) -> None:
    scores = {0: 5, 1: 2, 2: 4}  # one outlier sample
    j = judge(tmp_path, lambda p, s: {"score": scores[s], "reason": f"sample {s}"})
    r = asyncio.run(judged.relevance(j, "q", "a", samples=3))
    assert r.score == 0.75  # median 4/5, not the mean
    assert r.spread == 0.75 and "3 samples" in r.note
    assert sorted(j.inner.samples_seen) == [0, 1, 2]  # type: ignore[attr-defined]


def test_refusal_majority_vote(tmp_path: Path) -> None:
    votes = {0: True, 1: False, 2: True}
    j = judge(tmp_path, lambda p, s: {"declined": votes[s]})
    assert asyncio.run(judged.is_refusal(j, "q", "a", samples=3)) is True


def test_samples_are_cached_separately(tmp_path: Path) -> None:
    fn = FnJudge(lambda p, s: {"score": 3, "reason": ""})
    asyncio.run(judged.relevance(CachedJudge(fn, tmp_path / "c"), "q", "a", samples=3))
    asyncio.run(judged.relevance(CachedJudge(fn, tmp_path / "c"), "q", "a", samples=3))
    assert len(fn.samples_seen) == 3  # second run fully cached, per sample


def test_samples_must_be_odd() -> None:
    with pytest.raises(ValueError):
        Config.model_validate({"target": {"type": "python", "callable": "a:b"}, "judge": {"samples": 2}})


# ── Calibration ──────────────────────────────────────────────────────


def test_kappa() -> None:
    assert cohen_kappa(tp=10, fp=0, tn=10, fn=0) == 1.0
    assert cohen_kappa(tp=10, fp=10, tn=0, fn=0) == 0.0  # always "yes": no better than chance
    assert cohen_kappa(tp=8, fp=2, tn=8, fn=2) == pytest.approx(0.6)


def test_builtin_set_is_balanced() -> None:
    cases = load_labels(None)
    for kind in ("faithfulness", "relevance", "correctness", "refusal"):
        labels = [c.label for c in cases if c.kind == kind]
        assert len(labels) == 8 and sum(labels) == 4, kind


def oracle(cases: Any) -> Any:
    """A judge that knows the right answer for every built-in case (found by its answer text)."""
    truth = {c.answer: c.label for c in cases}

    def fn(prompt: str, sample: int) -> dict[str, Any]:
        label = next(v for a, v in truth.items() if a in prompt)
        if "Split the answer" in prompt:
            ok = {"claim": "a", "supported": True, "evidence": quote_from_prompt(prompt)}
            return {"claims": [ok] if label else [ok, {"claim": "b", "supported": False, "evidence": ""}]}
        if "decline" in prompt:
            return {"declined": label}
        return {"score": 5 if label else 1, "reason": ""}

    return fn


def cfg() -> Config:
    return Config.model_validate({"target": {"type": "python", "callable": "a:b"}})


def test_perfect_judge_calibrates(tmp_path: Path) -> None:
    cases = load_labels(None)
    results = asyncio.run(calibrate(cases, judge(tmp_path, oracle(cases)), cfg()))
    assert all(r.accuracy == 1.0 and r.kappa == 1.0 for r in results.values())


def test_lazy_judge_is_caught(tmp_path: Path) -> None:
    cases = load_labels(None)
    yes = lambda p, s: {"score": 5, "reason": "", "declined": True}  # noqa: E731
    results = asyncio.run(calibrate(cases, judge(tmp_path, yes), cfg()))
    assert results["relevance"].accuracy == 0.5 and results["relevance"].kappa == 0.0


def test_cutoffs_match_metric_meaning() -> None:
    assert CUTOFFS == {"faithfulness": 0.999, "relevance": 0.5, "correctness": 0.75}


def test_calibrate_cli_with_custom_labels(project: Path) -> None:
    (project / "labels.jsonl").write_text(
        json.dumps({"id": "x", "kind": "relevance", "question": "q", "answer": "a", "label": True}),
        encoding="utf-8",
    )
    r = CliRunner().invoke(app, ["calibrate", "--labels", "labels.jsonl"])
    assert r.exit_code == 2 and "needs a judge" in r.output  # project config uses judge: none


# ── Confidence intervals ─────────────────────────────────────────────


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean() -> None:
    diffs = [-0.25, 0.0, -0.5, 0.0, -0.25, 0.25, -0.5, 0.0]
    lo, hi = confidence_interval(diffs)
    assert (lo, hi) == confidence_interval(diffs)
    assert lo < sum(diffs) / len(diffs) < hi


def test_large_sets_use_normal_approximation_quickly() -> None:
    import time

    diffs = [(-1) ** i * 0.1 for i in range(50_000)]
    start = time.perf_counter()
    lo, hi = confidence_interval(diffs)
    assert time.perf_counter() - start < 0.5
    assert lo < 0 < hi


def mk(rid: str, rows: dict[str, float]) -> RunResult:
    results = [
        CaseResult(case=GoldenCase(id=i, question=i), scores={"relevance": s}) for i, s in rows.items()
    ]
    m, g = aggregate(results)
    return RunResult(id=rid, created_at="", results=results, metrics=m, guardrails=g)


def test_noisy_drop_is_only_a_warning_with_significance() -> None:
    base = mk("b", {f"q{i}": 0.75 for i in range(10)})
    # Mean drops 0.075, but half the questions improved: the interval includes zero
    cur = mk("c", {f"q{i}": (1.0 if i % 2 else 0.35) for i in range(10)})
    t = Thresholds(default_max_drop=0.05, require_significance=True)
    v = evaluate(cur, base, t)
    assert v.passed and any("not significantly" in w for w in v.warnings)
    assert not evaluate(cur, base, Thresholds(default_max_drop=0.05)).passed


def test_consistent_drop_fails_even_with_significance() -> None:
    base = mk("b", {f"q{i}": 1.0 for i in range(30)})
    cur = mk("c", {f"q{i}": 0.75 for i in range(30)})
    v = evaluate(cur, base, Thresholds(require_significance=True))
    assert not v.passed and "95% CI" in v.failures[0]
