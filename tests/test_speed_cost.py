"""Speed and cost against the baseline: shown when measured, hidden (not zero) when not, gated only when asked."""

from __future__ import annotations

import asyncio

import pytest

from answergate.baseline import SpeedCost, evaluate
from answergate.config import Config, Thresholds
from answergate.models import CaseResult, GoldenCase, RunResult, TargetResponse
from answergate.report import markdown, speed_line
from answergate.runner import build_run, run
from answergate.targets.parse import to_response


def _run(p50: float, p95: float, cost: float | None = None, tokens: float | None = None) -> RunResult:
    results = [
        CaseResult(case=GoldenCase(id=f"q{i}", question="q"), scores={"hit_rate": 1.0}) for i in range(25)
    ]
    r = build_run(results, "", "r", "exact")
    r.latency_p50_ms, r.latency_p95_ms = p50, p95
    r.cost_per_question, r.tokens_per_question, r.cost_currency = (
        cost,
        tokens,
        "$" if cost is not None else "",
    )
    return r


def test_latency_and_cost_changes_are_reported() -> None:
    v = evaluate(_run(2100, 3000, cost=0.0004), _run(4476, 5642, cost=0.0021), Thresholds())
    assert v.passed  # reported, never gated by default
    line = speed_line(v.speed)
    assert "Median answer time 4,476 ms → 2,100 ms (−53%)" in line
    assert "p95 5,642 ms → 3,000 ms (−47%)" in line
    assert "cost per question $0.002100 → $0.000400 (−81%)" in line
    assert "**Speed and cost:** Median answer time" in markdown(_run(2100, 3000, cost=0.0004), v)


def test_tokens_are_shown_when_there_are_no_prices() -> None:
    v = evaluate(_run(100, 200, tokens=900), _run(100, 200, tokens=1200), Thresholds())
    assert "tokens per question 1,200 → 900 (−25%)" in speed_line(v.speed)


def test_unmeasured_cost_is_hidden_not_zero() -> None:
    v = evaluate(_run(100, 200), _run(100, 200), Thresholds())
    line = speed_line(v.speed)
    assert "cost" not in line and "tokens" not in line and "(±0%)" in line


def test_no_baseline_no_speed_section() -> None:
    v = evaluate(_run(100, 200), None, Thresholds())
    assert v.speed is None and "Speed and cost" not in markdown(_run(100, 200), v)


def test_a_run_where_nothing_answered_has_no_latency_to_compare() -> None:
    v = evaluate(_run(0, 0), _run(100, 200), Thresholds())
    assert v.speed is not None and v.speed.p50_after is None
    assert "Median answer time" not in speed_line(v.speed)


@pytest.mark.parametrize(
    ("limit", "after_ms", "fails"),
    [(None, 9000, False), (50, 140, False), (50, 160, True)],
)
def test_latency_gate_only_when_set(limit: float | None, after_ms: float, fails: bool) -> None:
    v = evaluate(_run(after_ms, after_ms), _run(100, 100), Thresholds(max_latency_increase_pct=limit))
    assert v.passed is not fails
    if fails:
        assert "Median latency rose 60%; allowed 50%" in v.failures


def test_cost_gate_warns_when_cost_was_not_measured() -> None:
    v = evaluate(_run(100, 100), _run(100, 100), Thresholds(max_cost_increase_pct=10))
    assert v.passed and any("cost per question" in w and "wasn't measured" in w for w in v.warnings)


def test_change_pct_handles_missing_and_zero() -> None:
    assert SpeedCost.change_pct(None, 5) is None
    assert SpeedCost.change_pct(0, 5) is None
    assert SpeedCost.change_pct(4, 5) == 25


# ── Token usage from the pipeline ─────────────────────────────────────


@pytest.mark.parametrize(
    "usage",
    [
        {"input": 120, "output": 30},
        {"prompt_tokens": 120, "completion_tokens": 30},  # OpenAI
        {"input_tokens": 120, "output_tokens": 30},  # Anthropic
        {"prompt_eval_count": 120, "eval_count": 30},  # Ollama
    ],
)
def test_every_common_usage_spelling_is_read(usage: dict[str, int]) -> None:
    r = to_response({"answer": "x", "usage": usage})
    assert (r.tokens_in, r.tokens_out) == (120, 30)


@pytest.mark.parametrize("junk", [None, "lots", [1, 2], {"input": "many"}, {"input": -5, "output": True}])
def test_odd_usage_values_are_ignored(junk: object) -> None:
    r = to_response({"answer": "x", "usage": junk})
    assert r.tokens_in is None and r.tokens_out is None


class Counted:
    async def ask(self, question: str) -> TargetResponse:
        return TargetResponse(answer="ok", tokens_in=1000, tokens_out=200)

    async def aclose(self) -> None:
        pass


def test_cost_per_question_from_the_configured_prices() -> None:
    cfg = Config.model_validate(
        {
            "target": {"type": "python", "callable": "x:y"},
            "judge": {"provider": "none"},
            "cost": {"input_per_1k": 0.5, "output_per_1k": 1.5, "currency": "₹"},
        }
    )
    result = asyncio.run(run(cfg, [GoldenCase(id="a", question="q")], Counted(), None))  # type: ignore[arg-type]
    # 1000 in at ₹0.5/1k + 200 out at ₹1.5/1k = ₹0.80
    assert result.results[0].cost == pytest.approx(0.8)
    assert result.cost_per_question == pytest.approx(0.8)
    assert result.tokens_per_question == 1200
    assert result.cost_currency == "₹"


def test_prices_do_not_change_the_scoring_fingerprint() -> None:
    base = {"target": {"type": "python", "callable": "x:y"}, "judge": {"provider": "none"}}
    priced = {**base, "cost": {"input_per_1k": 1, "output_per_1k": 2}}
    assert Config.model_validate(base).fingerprint() == Config.model_validate(priced).fingerprint()
