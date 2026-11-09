"""`numbers`: do the figures in the answer match the figures in the reference answer?

An LLM judge can call "the rate is 12%" mostly correct when the reference says 18%: the words agree, the fact
doesn't. This metric needs no LLM. It reads the figures out of both texts (percentages, rupee amounts with
lakh/crore, plain numbers, ordinals like "20th") in any common spelling, and scores the share of the
reference's figures that the answer states.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["percent", "money", "plain"]

_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "twenty-eight": 28,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}  # fmt: skip
# A number word only counts in front of a unit, so "one of the" never becomes a figure
_UNIT = r"(?:%|per\s*cent|percent|lakhs?|lacs?|crores?|thousand|days?|months?|years?)"
_WORD_NUMBER = re.compile(
    r"\b(" + "|".join(sorted(_WORDS, key=len, reverse=True)) + r")\b(?=\s*" + _UNIT + ")", re.I
)

_MULTIPLIER = {
    "lakh": 1e5,
    "lakhs": 1e5,
    "lac": 1e5,
    "lacs": 1e5,
    "crore": 1e7,
    "crores": 1e7,
    "thousand": 1e3,
}
_FIGURE = re.compile(
    r"(?P<cur>₹|\brs\.?|\binr)?\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?:st|nd|rd|th)?\b"
    r"(?:\s*(?P<mult>lakhs?|lacs?|crores?|thousand)\b)?"
    r"(?P<pct>\s*(?:%|per\s*cent\b|percent\b))?",
    re.I,
)


# Text right before a number that makes it part of a name: a law section, a rule, a form or a return
_NAMED = re.compile(
    r"(?:\b(?:section|sec\.|rule|form|notification|no\.|article|schedule)\s*|[a-z]-|\d\(|\d/)\s*$", re.I
)


@dataclass(frozen=True)
class Figure:
    kind: Kind
    value: float
    text: str

    def same_value(self, other: Figure) -> bool:
        return abs(self.value - other.value) <= 1e-9 * max(1.0, abs(self.value))


def extract(text: str) -> list[Figure]:
    """Every figure in the text, normalised: "₹20 lakh", "Rs 20,00,000" and "2000000" all have value 2,000,000."""
    text = _WORD_NUMBER.sub(lambda m: str(_WORDS[m.group(1).lower()]), text)
    out: list[Figure] = []
    for m in _FIGURE.finditer(text):
        if not m.group("cur") and _NAMED.search(text[max(0, m.start() - 16) : m.start("num")]):
            continue  # "Section 16", "Rule 36(4)", "GSTR-1": part of a name, not a figure to check
        digits = m.group("num").replace(",", "")
        try:
            value = float(digits)
        except ValueError:
            continue
        mult = (m.group("mult") or "").lower()
        if mult:
            value *= _MULTIPLIER[mult]
        if m.group("pct"):
            kind: Kind = "percent"
        elif m.group("cur") or mult in ("lakh", "lakhs", "lac", "lacs", "crore", "crores"):
            kind = "money"
        else:
            kind = "plain"
        out.append(Figure(kind, value, m.group(0).strip()))
    return out


def _matches(ref: Figure, got: Figure) -> bool:
    # A plain number on either side matches any kind ("18" for "18%"); a percent never matches an amount
    if not ref.same_value(got):
        return False
    return ref.kind == got.kind or "plain" in (ref.kind, got.kind)


def score_numbers(answer: str, reference: str) -> tuple[float, str] | None:
    """(score, note), or None when the reference states no figure (nothing to check).

    Each figure in the reference counts 1 when the answer states it, 0 when it doesn't. A figure the answer
    hedges, by also giving a different percentage or amount the reference never mentions ("12% or 18%"),
    counts 0.5: a reader can't tell which one is right.
    """
    wanted: list[Figure] = []
    for f in extract(reference):
        if not any(f.same_value(w) and f.kind == w.kind for w in wanted):
            wanted.append(f)
    if not wanted:
        return None
    got = extract(answer)
    stray = [g for g in got if g.kind != "plain" and not any(_matches(w, g) for w in wanted)]
    credit: list[float] = []
    missing: list[str] = []
    hedged: list[str] = []
    for w in wanted:
        if not any(_matches(w, g) for g in got):
            credit.append(0.0)
            missing.append(w.text)
        elif any(s.kind == w.kind for s in stray):
            credit.append(0.5)
            hedged.append(w.text)
        else:
            credit.append(1.0)
    score = round(sum(credit) / len(credit), 4)
    if not missing and not hedged:
        return score, f"All {len(wanted)} figure(s) match: " + ", ".join(w.text for w in wanted)
    parts = []
    if missing:
        said = [g.text for g in got if g.kind in {w.kind for w in wanted} or g.kind == "plain"]
        parts.append(f"expected {', '.join(missing)}; answer says {', '.join(said) if said else 'no figure'}")
    if hedged:
        others = ", ".join(s.text for s in stray)
        parts.append(f"{', '.join(hedged)} stated alongside {others}")
    return score, "; ".join(parts)
