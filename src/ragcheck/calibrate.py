"""Measure the judge against labelled cases, so you know how far to trust its scores.

Each labelled case has a yes/no truth ("this answer is fully supported", "this answer is relevant",
"this answer matches the reference", "this answer declines"). The judge's score is turned into a yes/no
the same way the metric is meant to be read, then compared with the label.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from ragcheck.config import Config, ConfigError
from ragcheck.judges import CachedJudge, FatalJudgeError, JudgeError
from ragcheck.metrics import judged
from ragcheck.models import Context

Kind = Literal["faithfulness", "relevance", "correctness", "refusal"]

# How a score becomes yes/no. Faithfulness: every claim supported. Relevance: at least "partly answers"
# (3/5). Correctness: at least 4/5, i.e. the facts that matter agree.
CUTOFFS: dict[str, float] = {"faithfulness": 0.999, "relevance": 0.5, "correctness": 0.75}


class LabelledCase(BaseModel):
    id: str
    kind: Kind
    label: bool
    question: str = ""
    answer: str
    context: str = ""
    reference: str = ""


@dataclass
class KindResult:
    kind: str
    n: int = 0
    correct: int = 0
    # Confusion counts: true/false positives/negatives
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    errors: int = 0
    misses: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def kappa(self) -> float:
        return cohen_kappa(self.tp, self.fp, self.tn, self.fn)


def cohen_kappa(tp: int, fp: int, tn: int, fn: int) -> float:
    """Agreement beyond chance. 1 = perfect, 0 = no better than guessing from the base rates."""
    n = tp + fp + tn + fn
    if n == 0:
        return 0.0
    observed = (tp + tn) / n
    expected = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def load_labels(path: Path | None) -> list[LabelledCase]:
    if path is None:
        text = (resources.files("ragcheck") / "templates" / "calibration.jsonl").read_text(encoding="utf-8")
        source = "built-in calibration set"
    else:
        if not path.exists():
            raise ConfigError(f"Labels file not found: {path}")
        text, source = path.read_text(encoding="utf-8-sig"), path.name
    cases = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            cases.append(LabelledCase.model_validate(json.loads(line)))
        except (ValueError, ValidationError) as e:
            raise ConfigError(f"{source} line {n}: {e}") from e
    if not cases:
        raise ConfigError(f"{source} has no labelled cases")
    return cases


async def _predict(case: LabelledCase, judge: CachedJudge, cfg: Config) -> bool:
    n = cfg.judge.samples
    if case.kind == "refusal":
        return await judged.is_refusal(judge, case.question, case.answer, n)
    if case.kind == "faithfulness":
        ctx = [Context(id="context", text=case.context)]
        result = await judged.faithfulness(judge, case.answer, ctx, n, cfg.judge.verify_evidence)
    elif case.kind == "relevance":
        result = await judged.relevance(judge, case.question, case.answer, n)
    else:
        result = await judged.correctness(judge, case.question, case.answer, case.reference, n)
    return result.score >= CUTOFFS[case.kind]


async def calibrate(cases: list[LabelledCase], judge: CachedJudge, cfg: Config) -> dict[str, KindResult]:
    async def one(case: LabelledCase) -> tuple[LabelledCase, bool | None]:
        try:
            return case, await _predict(case, judge, cfg)
        except FatalJudgeError:
            raise
        except JudgeError:
            return case, None

    results: dict[str, KindResult] = {}
    for case, predicted in await asyncio.gather(*(one(c) for c in cases)):
        r = results.setdefault(case.kind, KindResult(case.kind))
        r.n += 1
        if predicted is None:
            # A failed judge call counts as wrong: an unreliable judge must not look accurate
            r.errors += 1
            r.misses.append(f"{case.id}: judge error")
            continue
        if predicted == case.label:
            r.correct += 1
        else:
            r.misses.append(f"{case.id}: label {case.label}, judge said {predicted}")
        if predicted and case.label:
            r.tp += 1
        elif predicted and not case.label:
            r.fp += 1
        elif not predicted and not case.label:
            r.tn += 1
        else:
            r.fn += 1
    return dict(sorted(results.items()))
