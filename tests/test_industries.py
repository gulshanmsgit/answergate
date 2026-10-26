"""The industry demo as a test: six pipelines, seven injected faults each, and the gate must catch every
fault that retrieval metrics and guardrails can see, with no false alarm on a healthy rerun."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples" / "industries"
sys.path.insert(0, str(EXAMPLES))

import run_demo  # noqa: E402
from demo_rag import FAULTS, INDUSTRIES, answer  # noqa: E402

from ragcheck.config import load_golden  # noqa: E402

# The metric or guardrail each fault must trip
TRIPS = {
    "stale_index": ("hit_rate", "context_precision", "context_recall", "mrr", "ndcg"),
    "top_k_1": ("context_recall", "ndcg"),
    "rerank_bug": ("mrr", "ndcg"),
    "no_refusal": ("'refusal'",),
    "pii_leak": ("'pii'",),
    "injectable": ("'injection'",),
}


@pytest.mark.parametrize("industry", INDUSTRIES)
def test_every_visible_fault_is_caught(
    industry: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEMO_FAULT", raising=False)
    cfg = run_demo.configure(industry, None, None, tmp_path)
    base = run_demo.check(industry, cfg, tmp_path, "baseline", "", False)
    assert base.passed, base.why
    rerun = run_demo.check(industry, cfg, tmp_path, "healthy-rerun", "", False)
    assert rerun.outcome == "PASS", rerun.why

    for fault in FAULTS:
        row = run_demo.check(industry, cfg, tmp_path, fault, fault, False)
        if fault == "hallucinate":
            # Invisible without a judge, and the demo says so instead of claiming a catch
            assert row.outcome == "NEEDS JUDGE", row.why
            continue
        assert row.outcome == "CAUGHT", f"{industry}/{fault}: {row.why}"
        assert any(t in row.why for t in TRIPS[fault]), (
            f"{industry}/{fault} tripped for another reason: {row.why}"
        )
    assert (tmp_path / industry / "pii_leak.html").exists()


@pytest.mark.parametrize("industry", INDUSTRIES)
def test_golden_sets_are_well_formed(industry: str) -> None:
    cases = load_golden(EXAMPLES / industry / "golden.jsonl")
    assert len(cases) >= 20, "fewer than 20 questions triggers RAGCheck's small-set warning"
    assert len({c.id for c in cases}) == len(cases)
    assert sum(c.expect_refusal for c in cases) >= 2
    assert sum(bool(c.canary) for c in cases) == 1
    assert sum(len(c.expected_sources) > 1 for c in cases) >= 3, (
        "multi-document questions catch top_k changes"
    )
    assert all(c.reference_answer for c in cases if not c.expect_refusal)


def test_unknown_fault_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_FAULT", "typo")
    with pytest.raises(ValueError, match="Unknown DEMO_FAULT"):
        answer("banking", "What is the savings interest rate?")
