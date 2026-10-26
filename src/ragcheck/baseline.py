"""Run history on disk, and the regression gate that compares a run with its baseline."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from ragcheck.config import Thresholds
from ragcheck.models import RunResult

# A single question counts as regressed when one of its scores falls by at least this much
CASE_DROP = 0.25


def save_run(run: RunResult, workdir: Path) -> Path:
    runs = workdir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{run.id}.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_run(path: Path) -> RunResult:
    return RunResult.model_validate_json(path.read_text(encoding="utf-8"))


def latest_run(workdir: Path) -> Path | None:
    runs = sorted((workdir / "runs").glob("*.json"))
    return runs[-1] if runs else None


def save_baseline(run: RunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")


@dataclass
class MetricDelta:
    name: str
    current: float | None
    baseline: float | None
    # Change measured only on questions present in both runs (None: plain difference of the means)
    paired: float | None = None
    paired_n: int = 0
    # 95% confidence interval of the paired change (None without a baseline)
    ci_low: float | None = None
    ci_high: float | None = None

    @property
    def delta(self) -> float | None:
        if self.paired is not None:
            return round(self.paired, 4)
        if self.current is None or self.baseline is None:
            return None
        return round(self.current - self.baseline, 4)


@dataclass
class CaseRegression:
    case_id: str
    question: str
    what: str
    before: str
    after: str


@dataclass
class Verdict:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: list[MetricDelta] = field(default_factory=list)
    guardrails: list[MetricDelta] = field(default_factory=list)
    regressions: list[CaseRegression] = field(default_factory=list)
    improvements: int = 0

    @property
    def regressed_cases(self) -> list[str]:
        return list(dict.fromkeys(r.case_id for r in self.regressions))


# Below this many questions, one question moves a mean by a noticeable step
SMALL_SET = 20
# Bootstrap below this many pairs; above it the normal approximation is accurate and much faster
BOOTSTRAP_MAX_N = 400
BOOTSTRAP_RESAMPLES = 2000


def _paired_diffs(current: RunResult, baseline: RunResult, name: str) -> list[float]:
    """Per-question change of a metric over the questions both runs scored, so adding or removing
    golden questions never shows up as a regression or hides one."""
    before = {r.case.id: r.scores[name] for r in baseline.results if name in r.scores}
    return [
        r.scores[name] - before[r.case.id]
        for r in current.results
        if name in r.scores and r.case.id in before
    ]


def confidence_interval(diffs: list[float]) -> tuple[float, float]:
    """95% interval for the mean paired change. Seeded bootstrap for small sets, so the same runs always
    give the same interval; normal approximation for large ones."""
    n = len(diffs)
    mean = sum(diffs) / n
    if n == 1:
        return mean, mean
    if n > BOOTSTRAP_MAX_N:
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        half = 1.96 * math.sqrt(var / n)
        return mean - half, mean + half
    rng = random.Random(0)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(BOOTSTRAP_RESAMPLES))
    return means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]


def evaluate(current: RunResult, baseline: RunResult | None, t: Thresholds) -> Verdict:
    v = Verdict(passed=True)
    base_m = baseline.metrics if baseline else {}
    base_g = baseline.guardrails if baseline else {}

    for name in sorted(set(current.metrics) | set(base_m)):
        d = MetricDelta(name, current.metrics.get(name), base_m.get(name))
        if baseline and d.current is not None and d.baseline is not None:
            diffs = _paired_diffs(current, baseline, name)
            if diffs:
                d.paired, d.paired_n = sum(diffs) / len(diffs), len(diffs)
                lo, hi = confidence_interval(diffs)
                d.ci_low, d.ci_high = round(lo, 4), round(hi, 4)
        v.metrics.append(d)
        floor = t.min.get(name)
        if floor is not None and d.current is not None and d.current < floor:
            v.failures.append(f"{name} is {d.current:.3f}, below the minimum {floor:.3f}")
        allowed = t.max_drop.get(name, t.default_max_drop)
        if allowed is not None and d.delta is not None and -d.delta > allowed:
            on = f" on the {d.paired_n} questions in both runs" if d.paired is not None else ""
            ci = f" (95% CI {d.ci_low:+.3f} to {d.ci_high:+.3f})" if d.ci_high is not None else ""
            if t.require_significance and d.ci_high is not None and d.ci_high >= 0:
                # The interval still includes "no change": too uncertain to block on
                v.warnings.append(f"{name} dropped {-d.delta:.3f}{on}, but not significantly{ci}")
            else:
                v.failures.append(f"{name} dropped {-d.delta:.3f}{on}; allowed {allowed:.3f}{ci}")
        if d.current is None and d.baseline is not None:
            # A metric silently vanishing (judge switched off, every judge call failing) must not pass
            v.failures.append(f"{name} was measured in the baseline but not in this run (is the judge off?)")

    for name in t.min:
        if name not in current.metrics:
            v.warnings.append(f"A minimum is set for {name}, but no case measured it")

    for name in sorted(set(current.guardrails) | set(base_g)):
        d = MetricDelta(name, current.guardrails.get(name), base_g.get(name))
        v.guardrails.append(d)
        if t.guardrails_must_pass and d.current is not None and d.current < 1.0:
            failing = [r.case.id for r in current.results if r.guardrails.get(name) is False]
            new = ""
            if t.guardrail_gate == "new" and baseline:
                known = {r.case.id for r in baseline.results if r.guardrails.get(name) is False}
                if still := sum(1 for cid in failing if cid in known):
                    v.warnings.append(
                        f"Guardrail '{name}' still fails on {still} known case(s) from the baseline"
                    )
                failing, new = [cid for cid in failing if cid not in known], "new "
            if failing:
                v.failures.append(f"Guardrail '{name}' failed on {len(failing)} {new}case(s)")

    if current.errors:
        v.failures.append(f"The pipeline failed on {current.errors} question(s); see the report for errors")
    if current.judge_errors:
        v.failures.append(f"The judge failed on {current.judge_errors} question(s), so scores are incomplete")

    if baseline:
        if baseline.config_hash and baseline.config_hash != current.config_hash:
            v.warnings.append(
                "Scoring settings (judge, metrics or k) differ from the baseline; deltas may not be like-for-like"
            )
        now_ids = {r.case.id for r in current.results}
        then_ids = {r.case.id for r in baseline.results}
        if added := len(now_ids - then_ids):
            v.warnings.append(f"{added} new question(s) not in the baseline: checked against minimums only")
        if removed := len(then_ids - now_ids):
            v.warnings.append(f"{removed} baseline question(s) are no longer in the golden set")
        _case_diffs(current, baseline, v)

    if 0 < len(current.results) < SMALL_SET:
        n = len(current.results)
        v.warnings.append(
            f"Only {n} questions: one question can move a metric by up to {1 / n:.2f}. Use 20+ for a stable gate."
        )

    v.passed = not v.failures
    return v


def _case_diffs(current: RunResult, baseline: RunResult, v: Verdict) -> None:
    before = {r.case.id: r for r in baseline.results}
    for r in current.results:
        b = before.get(r.case.id)
        if b is None:
            continue
        for name, score in r.scores.items():
            old = b.scores.get(name)
            if old is None:
                continue
            if old - score >= CASE_DROP:
                v.regressions.append(
                    CaseRegression(r.case.id, r.case.question, name, f"{old:.2f}", f"{score:.2f}")
                )
            elif score - old >= CASE_DROP:
                v.improvements += 1
        for name, ok in r.guardrails.items():
            if b.guardrails.get(name) is True and ok is False:
                v.regressions.append(
                    CaseRegression(r.case.id, r.case.question, f"guardrail:{name}", "pass", "fail")
                )
        if b.error is None and r.error is not None:
            v.regressions.append(CaseRegression(r.case.id, r.case.question, "error", "ok", r.error[:80]))
