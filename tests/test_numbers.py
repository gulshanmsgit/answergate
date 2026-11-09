"""The `numbers` metric: an answer must state the reference's figures, in whatever spelling.

Two kinds of test, as for consistency: it MUST fail a wrong rate, limit or date, and MUST NOT fail the same
figure written differently.
"""

from __future__ import annotations

import asyncio

import pytest

from answergate.config import Config
from answergate.metrics.numbers import extract, score_numbers
from answergate.models import GoldenCase, TargetResponse
from answergate.report import markdown
from answergate.runner import run


def score(answer: str, reference: str) -> float:
    result = score_numbers(answer, reference)
    assert result is not None
    return result[0]


# ── Must NOT fail: the same figure, spelled differently ───────────────


@pytest.mark.parametrize(
    ("answer", "reference"),
    [
        ("GST on this is 18 %.", "The rate is 18%."),
        ("It attracts eighteen per cent tax.", "The rate is 18%."),
        ("Tax at 18 percent applies.", "The rate is 18%."),
        ("Register once turnover crosses Rs 20,00,000.", "The threshold is ₹20 lakh."),
        ("The limit is 2,000,000 rupees.", "The threshold is ₹20 lakh."),
        ("Above INR 20 lakhs you must register.", "The threshold is ₹20 lakh."),
        ("₹1.5 crore is the cap.", "The composition limit is Rs 150 lakh."),
        ("File it by the 20th.", "GSTR-3B is due on the 20th of the following month."),
        ("A late fee of ₹50 a day applies.", "The late fee is Rs. 50 per day."),
    ],
)
def test_the_same_figure_in_another_spelling_matches(answer: str, reference: str) -> None:
    assert score(answer, reference) == 1.0


def test_numbers_inside_names_are_not_figures_to_check() -> None:
    # Section numbers, rules and form names are names, not facts the answer must repeat
    reference = "Under Section 16(4) and Rule 36, input tax credit on GSTR-2B is limited to 105%."
    assert [f.value for f in extract(reference)] == [105.0]
    assert score("ITC is capped at 105% of what GSTR-2B shows.", reference) == 1.0


def test_extra_unrelated_numbers_do_not_cost_anything() -> None:
    # Mentioning a date or a count alongside the right rate is not hedging the rate
    assert score("Since 1 July 2017, the rate has been 18% for these 3 services.", "The rate is 18%.") == 1.0


# ── Must FAIL: a wrong rate, limit or date ────────────────────────────


@pytest.mark.parametrize(
    ("answer", "reference"),
    [
        ("The GST rate is 12%.", "The GST rate is 18%."),
        ("You must register above ₹40 lakh.", "Service providers must register above ₹20 lakh."),
        ("GSTR-3B is due on the 11th.", "GSTR-3B is due on the 20th of the following month."),
        ("Interest is charged at 24% a year.", "Interest is 18% per annum."),
        ("The late fee is ₹100 per day.", "The late fee is ₹50 per day."),
        ("I don't know based on the documents.", "The rate is 18%."),
    ],
)
def test_a_changed_figure_scores_zero(answer: str, reference: str) -> None:
    assert score(answer, reference) == 0.0


def test_hedging_between_two_rates_gets_half() -> None:
    # "12% or 18%" contains the right rate, but a reader can't tell which one applies
    result = score_numbers("It is either 12% or 18%.", "The rate is 18%.")
    assert result is not None and result[0] == 0.5
    assert "alongside 12%" in result[1]


def test_a_percentage_never_matches_an_amount() -> None:
    assert score("Pay ₹18.", "The rate is 18%.") == 0.0


def test_partial_credit_when_some_figures_are_right() -> None:
    reference = "Interest is 18% and the late fee is ₹50 per day."
    assert score("Interest is 18%; the late fee is ₹100 per day.", reference) == 0.5


def test_the_note_says_what_was_expected_and_what_was_said() -> None:
    result = score_numbers("The rate is 12%.", "The rate is 18%.")
    assert result is not None and result[1] == "expected 18%; answer says 12%"


def test_no_figure_in_the_reference_means_not_scored() -> None:
    assert score_numbers("Anything at all, 42%.", "Explain how input tax credit works.") is None


# ── Inside a run ──────────────────────────────────────────────────────


class Fixed:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def ask(self, question: str) -> TargetResponse:
        return TargetResponse(answer=self.answer)

    async def aclose(self) -> None:
        pass


def _cfg() -> Config:
    return Config.model_validate(
        {"target": {"type": "python", "callable": "x:y"}, "judge": {"provider": "none"}}
    )


def test_numbers_is_scored_without_a_judge_and_skips_refusal_questions() -> None:
    cases = [
        GoldenCase(id="rate", question="GST rate?", reference_answer="The rate is 18%."),
        GoldenCase(id="no-ref", question="Explain ITC."),
        GoldenCase(
            id="decline", question="How do I evade GST?", reference_answer="Decline: 0%.", expect_refusal=True
        ),
    ]
    result = asyncio.run(run(_cfg(), cases, Fixed("It is 12%."), None))  # type: ignore[arg-type]
    by_id = {r.case.id: r for r in result.results}
    assert by_id["rate"].scores["numbers"] == 0.0
    assert by_id["rate"].notes["numbers"] == "expected 18%; answer says 12%"
    assert "numbers" not in by_id["no-ref"].scores
    assert "numbers" not in by_id["decline"].scores
    assert result.metrics["numbers"] == 0.0


def test_a_wrong_figure_fails_the_gate_and_is_named_in_the_comment() -> None:
    from answergate.baseline import evaluate
    from answergate.config import Thresholds

    cases = [GoldenCase(id=f"q{i}", question="q", reference_answer="The rate is 18%.") for i in range(25)]
    good = asyncio.run(run(_cfg(), cases, Fixed("It is 18%."), None))  # type: ignore[arg-type]
    bad = asyncio.run(run(_cfg(), cases, Fixed("It is 12%."), None))  # type: ignore[arg-type]
    v = evaluate(bad, good, Thresholds())
    assert not v.passed and any(f.startswith("numbers dropped") for f in v.failures)
    assert "Numbers correct dropped 100%" in markdown(bad, v)
