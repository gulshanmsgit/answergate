"""Trying to break pipeline-change detection and answer consistency.

Two kinds of test: the gate MUST fail on a real problem, and MUST NOT fail on a harmless difference.
The end-to-end tests drive the real CLI against a fake pipeline whose temperature and stability we control.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from answergate.baseline import evaluate
from answergate.cli import app
from answergate.config import Config, Thresholds
from answergate.models import CaseResult, GoldenCase, RunResult, TargetResponse
from answergate.report import markdown
from answergate.runner import answer_overlap, build_run, consistency, run
from answergate.targets.parse import to_response

# ── Must FAIL: instability that word overlap alone would miss ────────


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("max_age defaults to 600 seconds.", "max_age defaults to 60 seconds."),  # a number flips
        ("GET is allowed by default.", "GET is not allowed by default."),  # a negation flips
        ("The refund window is 30 days.", "The refund window is 30.5 days."),
        ("You can use allow_credentials with wildcards.", "You cannot use allow_credentials with wildcards."),
    ],
)
def test_contradicting_answers_count_as_at_most_half_consistent(a: str, b: str) -> None:
    assert answer_overlap(a, b) <= 0.5


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # The tricky word is in one answer only: a negation check without word boundaries would see
        # "no" in "know" (or "not" in "nothing") on one side and wrongly call the pair contradictory
        ("I know the default is GET.", "The default is GET."),
        ("Nothing else is needed: just GET.", "Just GET is needed."),
        ("Now set allow_origins to the list.", "Set allow_origins to the list."),
        ("Note: the default is GET.", "The default is GET."),
    ],
)
def test_words_containing_no_or_not_are_not_negations(a: str, b: str) -> None:
    assert answer_overlap(a, b) > 0.5


def test_a_sometimes_i_dont_know_answer_is_inconsistent() -> None:
    assert consistency(["Only GET.", "Only GET.", "I don't know based on the FastAPI docs."]) < 0.5


def _runs(
    consistency_score: float, rid: str, n: int = 30, pipeline: dict[str, str] | None = None
) -> RunResult:
    results = [
        CaseResult(case=GoldenCase(id=f"q{i}", question="q"), scores={"consistency": consistency_score})
        for i in range(n)
    ]
    r = build_run(results, "", rid, "exact")
    r.pipeline = pipeline or {}
    return r


def test_lowering_repeats_below_the_baseline_fails_with_a_clear_reason() -> None:
    no_consistency = build_run(
        [CaseResult(case=GoldenCase(id=f"q{i}", question="q"), scores={}) for i in range(30)],
        "",
        "c",
        "exact",
    )
    v = evaluate(no_consistency, _runs(1.0, "b"), Thresholds())
    assert not v.passed
    assert any("consistency was measured in the baseline" in f and "repeats" in f for f in v.failures)


def test_a_small_drop_within_the_margin_passes_but_a_big_one_fails() -> None:
    assert evaluate(_runs(0.97, "c"), _runs(1.0, "b"), Thresholds()).passed
    assert not evaluate(_runs(0.90, "c"), _runs(1.0, "b"), Thresholds()).passed


# ── Must NOT cry wolf ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ({"temperature": 0}, {"temperature": 0.0}),
        ({"top_k": 4}, {"top_k": 4.0}),
        ({"stream": True}, {"stream": True}),
        ({"llm": {"model": "a", "t": 1}}, {"llm": {"t": 1.0, "model": "a"}}),  # key order, int vs float
        ({"model": "granite4.1:8b"}, {"model": "granite4.1:8b  "}),  # stray whitespace
    ],
)
def test_same_settings_spelled_differently_are_not_a_change(before: dict, after: dict) -> None:
    b = _runs(1.0, "b", pipeline=to_response({"answer": "x", "pipeline": before}).pipeline)
    c = _runs(1.0, "c", pipeline=to_response({"answer": "x", "pipeline": after}).pipeline)
    assert evaluate(c, b, Thresholds()).pipeline_changes == []


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Only GET is allowed by default.", "only get is allowed by default"),
        ("Only GET is allowed by default.", "  Only GET   is allowed by default!  "),
    ],
)
def test_case_punctuation_and_spacing_do_not_lower_consistency(a: str, b: str) -> None:
    assert answer_overlap(a, b) == 1.0


def test_a_changed_setting_with_unchanged_quality_passes_and_is_still_reported() -> None:
    v = evaluate(_runs(1.0, "c", pipeline={"llm": "b"}), _runs(1.0, "b", pipeline={"llm": "a"}), Thresholds())
    assert v.passed and [(c.key, c.before, c.after) for c in v.pipeline_changes] == [("llm", "a", "b")]


# ── Garbage in ───────────────────────────────────────────────────────


@pytest.mark.parametrize("junk", [None, "a string", 42, [1, 2], {"deep": {"er": {"est": [1, {"x": None}]}}}])
def test_odd_pipeline_values_never_crash(junk: object) -> None:
    r = to_response({"answer": "x", "pipeline": junk})
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in r.pipeline.items())


def test_settings_with_pipes_newlines_and_many_keys_keep_the_comment_readable() -> None:
    # Pipes on both sides: escaping only one side would still break the comment. The key sorts first,
    # so it is among the 10 changes shown (a version of this test once never displayed it at all)
    before = {f"k{i}": "a" for i in range(15)} | {"a_prompt": "line one | x"}
    after = {f"k{i}": "b" for i in range(15)} | {"a_prompt": "line one\nline | two"}
    v = evaluate(_runs(1.0, "c", pipeline=after), _runs(1.0, "b", pipeline=before), Thresholds())
    md = markdown(_runs(1.0, "c", pipeline=after), v)
    line = next(x for x in md.splitlines() if x.startswith("**What changed"))
    assert "line one" in line and "and 6 more" in line and "\n" not in line
    assert line.count("|") == line.count("\\|") == 2  # both pipes shown, both escaped


class _FailsOnRepeat:
    """Answers the first ask, then times out on the repeats."""

    def __init__(self) -> None:
        self.calls = 0

    async def ask(self, question: str) -> TargetResponse:
        self.calls += 1
        if self.calls > 1:
            raise TimeoutError("model overloaded")
        return TargetResponse(answer="Only GET.")

    async def aclose(self) -> None:
        pass


def test_a_failed_repeat_is_not_scored_as_consistent() -> None:
    cfg = Config.model_validate(
        {
            "target": {"type": "python", "callable": "x:y"},
            "judge": {"provider": "none"},
            "repeats": 3,
            "retries": 0,
        }
    )
    result = asyncio.run(run(cfg, [GoldenCase(id="q", question="q")], _FailsOnRepeat(), None))  # type: ignore[arg-type]
    r = result.results[0]
    assert "consistency" not in r.scores  # never a silent 1.0
    assert r.notes["consistency"].startswith("Not scored: a repeat failed")


# ── End to end through the CLI ───────────────────────────────────────

FAKE = """
import itertools, os
ANSWERS = {
    "max_age": ["max_age defaults to 600 seconds.", "max_age defaults to 60 seconds.", "I don't know."],
    "methods": ["Only GET is allowed by default.", "GET is not allowed by default.", "Only GET."],
}
_turn = itertools.count()

def answer(question):
    temperature = float(os.environ.get("FAKE_TEMPERATURE", "0"))
    key = "max_age" if "max_age" in question else "methods"
    # At temperature 0 the first answer every time; above 0 a different answer on each call
    i = next(_turn) % 3 if temperature > 0 else 0
    return {
        "answer": ANSWERS[key][i],
        "contexts": [{"id": key + ".md#0", "text": ANSWERS[key][0]}],
        "pipeline": {"llm": "fake-8b", "temperature": temperature},
    }
"""


@pytest.fixture
def unstable_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "fake_llm.py").write_text(FAKE, encoding="utf-8")
    golden = [
        {
            "id": f"max-age-{i}",
            "question": f"What does max_age default to? ({i})",
            "expected_sources": ["max_age.md"],
        }
        for i in range(12)
    ] + [
        {
            "id": f"methods-{i}",
            "question": f"Which methods are allowed? ({i})",
            "expected_sources": ["methods.md"],
        }
        for i in range(12)
    ]
    (tmp_path / "golden.jsonl").write_text("\n".join(json.dumps(g) for g in golden), encoding="utf-8")
    (tmp_path / "answergate.yaml").write_text(
        "target: { type: python, callable: fake_llm:answer }\njudge: { provider: none }\nrepeats: 3\nconcurrency: 1\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_raising_the_temperature_fails_the_gate_and_says_why(
    unstable_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.setenv("FAKE_TEMPERATURE", "0")
    first = runner.invoke(app, ["run", "--save-baseline"])
    assert first.exit_code == 0, first.output

    monkeypatch.setenv("FAKE_TEMPERATURE", "1.0")
    second = runner.invoke(app, ["run", "--md", "summary.md"])
    assert second.exit_code == 1, second.output
    assert "temperature: 0 → 1" in second.output
    assert "consistency dropped" in second.output
    md = (unstable_project / "summary.md").read_text(encoding="utf-8")
    assert "**What changed in the pipeline:** `temperature` 0 → 1" in md
    assert "Answer consistency dropped" in md


def test_the_same_temperature_run_twice_passes(
    unstable_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    monkeypatch.setenv("FAKE_TEMPERATURE", "0")
    assert runner.invoke(app, ["run", "--save-baseline"]).exit_code == 0
    again = runner.invoke(app, ["run"])
    assert again.exit_code == 0, again.output
    assert "Pipeline changes" not in again.output


# ── A broken run must never become the baseline ──────────────────────


def test_a_run_where_every_question_failed_is_refused_as_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What actually happened once: the pipeline died mid-experiment, every question errored, and the
    # empty run was accepted as the baseline, so no later regression could ever fail the gate
    (tmp_path / "dead.py").write_text(
        "def answer(q):\n    raise ConnectionError('model server down')\n", encoding="utf-8"
    )
    (tmp_path / "golden.jsonl").write_text(
        "\n".join(
            json.dumps({"id": f"q{i}", "question": f"q{i}?", "expected_sources": ["a.md"]}) for i in range(3)
        ),
        encoding="utf-8",
    )
    (tmp_path / "answergate.yaml").write_text(
        "target: { type: python, callable: dead:answer }\njudge: { provider: none }\nretries: 0\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert runner.invoke(app, ["run"]).exit_code == 1
    refused = runner.invoke(app, ["baseline"])
    assert refused.exit_code == 2, refused.output
    assert "3 of 3 questions got no answer" in refused.output and "ConnectionError" in refused.output
    assert not (tmp_path / ".answergate" / "baseline.json").exists()
    forced = runner.invoke(app, ["baseline", "--force"])
    assert forced.exit_code == 0 and (tmp_path / ".answergate" / "baseline.json").exists()


def test_a_clean_failing_run_can_still_be_accepted_as_baseline(unstable_project: Path) -> None:
    # Accepting a run that fails the gate on purpose (known guardrail failures) must keep working
    runner = CliRunner()
    assert runner.invoke(app, ["run"]).exit_code in (0, 1)
    ok = runner.invoke(app, ["baseline"])
    assert ok.exit_code == 0, ok.output
