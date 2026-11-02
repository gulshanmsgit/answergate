"""Pipeline settings ("what changed") and answer consistency across repeated asks."""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

import pytest

from answergate.baseline import evaluate
from answergate.config import Config, Thresholds
from answergate.models import CaseResult, GoldenCase, RunResult, TargetResponse
from answergate.report import markdown
from answergate.runner import answer_overlap, build_run, consistency, run, run_pipeline_settings
from answergate.targets.parse import flatten, to_response


def test_pipeline_settings_are_read_and_flattened() -> None:
    r = to_response(
        {
            "answer": "Only GET.",
            "contexts": [],
            "pipeline": {"llm": {"model": "granite4.1:8b", "temperature": 0}, "chunk_size": 1000},
        }
    )
    assert r.pipeline == {"llm.model": "granite4.1:8b", "llm.temperature": "0", "chunk_size": "1000"}
    assert to_response({"answer": "a"}).pipeline == {}  # optional
    assert flatten("x") == {}


def test_run_settings_take_the_common_value_and_flag_variation() -> None:
    def res(model: str, t: str) -> CaseResult:
        return CaseResult(
            case=GoldenCase(id=model + t, question="q"), pipeline={"model": model, "temperature": t}
        )

    settings = run_pipeline_settings([res("a", "0"), res("a", "0"), res("b", "0")])
    assert settings == {"model": "a (varied: 2 values)", "temperature": "0"}


def _run(rid: str, pipeline: dict[str, str], score: float = 1.0) -> RunResult:
    results = [
        CaseResult(case=GoldenCase(id=f"q{i}", question="q"), scores={"correctness": score})
        for i in range(25)
    ]
    r = build_run(results, "", rid, "exact")
    r.pipeline = pipeline
    return r


def test_changed_settings_are_reported_against_the_baseline() -> None:
    base = _run("b", {"llm": "granite4.1:8b", "temperature": "0", "top_k": "4"})
    cur = _run("c", {"llm": "granite4.1:8b", "temperature": "1.0", "chunk_size": "250"}, score=0.8)
    v = evaluate(cur, base, Thresholds())
    assert [(c.key, c.before, c.after) for c in v.pipeline_changes] == [
        ("chunk_size", "", "250"),
        ("temperature", "0", "1.0"),
        ("top_k", "4", ""),
    ]
    md = markdown(cur, v)
    assert "**What changed in the pipeline:** `chunk_size` not reported → 250 · `temperature` 0 → 1.0" in md


def test_no_settings_no_change_section() -> None:
    v = evaluate(_run("c", {}), _run("b", {"llm": "x"}), Thresholds())
    assert v.pipeline_changes == [] and "What changed" not in markdown(_run("c", {}), v)


def test_answer_overlap_and_consistency() -> None:
    assert answer_overlap("Only GET is allowed.", "only get is allowed") == 1.0
    assert answer_overlap("Only GET.", "I don't know.") == 0.0
    assert consistency(["a b c", "a b c", "a b c"]) == 1.0
    mixed = consistency(["The default is GET.", "GET is the default.", "I don't know based on the docs."])
    assert 0.2 < mixed < 0.7
    assert consistency(["only one"]) == 1.0


class Flaky:
    """Gives a different answer on each call, like a high-temperature model."""

    def __init__(self, answers: list[str]) -> None:
        self.cycle = itertools.cycle(answers)
        self.calls = 0

    async def ask(self, question: str) -> TargetResponse:
        self.calls += 1
        return TargetResponse(answer=next(self.cycle), pipeline={"temperature": "1.0"})

    async def aclose(self) -> None:
        pass


def _cfg(**kw: Any) -> Config:
    return Config.model_validate(
        {"target": {"type": "python", "callable": "x:y"}, "judge": {"provider": "none"}, **kw}
    )


def test_repeats_score_consistency_and_keep_the_other_answers() -> None:
    cases = [GoldenCase(id="cors", question="Which methods are allowed by default?")]
    target = Flaky(["Only GET is allowed by default.", "I don't know based on the FastAPI docs."])
    result = asyncio.run(run(_cfg(repeats=3), cases, target, None))  # type: ignore[arg-type]
    r = result.results[0]
    assert target.calls == 3 and len(r.repeat_answers) == 2
    assert 0 < r.scores["consistency"] < 1
    assert "share" in r.notes["consistency"]
    assert result.metrics["consistency"] == r.scores["consistency"]
    assert result.pipeline == {"temperature": "1.0"}


def test_one_ask_by_default_and_none_for_refusal_questions() -> None:
    cases = [
        GoldenCase(id="a", question="q"),
        GoldenCase(id="weather", question="Weather tomorrow?", expect_refusal=True),
    ]
    once = Flaky(["x"])
    asyncio.run(run(_cfg(), cases[:1], once, None))  # type: ignore[arg-type]
    assert once.calls == 1
    target = Flaky(["same answer"])
    result = asyncio.run(run(_cfg(repeats=3), cases, target, None))  # type: ignore[arg-type]
    assert target.calls == 4  # 3 for the answerable question, 1 for the refusal question
    assert result.results[0].scores["consistency"] == 1.0
    assert "consistency" not in result.results[1].scores


def test_a_drop_in_consistency_fails_the_gate() -> None:
    def runs(score: float, rid: str) -> RunResult:
        results = [
            CaseResult(case=GoldenCase(id=f"q{i}", question="q"), scores={"consistency": score})
            for i in range(25)
        ]
        return build_run(results, "", rid, "exact")

    v = evaluate(runs(0.6, "c"), runs(1.0, "b"), Thresholds())
    assert not v.passed and any(f.startswith("consistency dropped") for f in v.failures)
    assert "Answer consistency dropped 40%" in markdown(runs(0.6, "c"), v)


@pytest.mark.parametrize("bad", [0, 11])
def test_repeats_bounds(bad: int) -> None:
    with pytest.raises(ValueError):
        _cfg(repeats=bad)
