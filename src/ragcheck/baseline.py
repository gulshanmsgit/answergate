"""Run history on disk, and the regression gate that compares a run with its baseline."""

from __future__ import annotations

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

    @property
    def delta(self) -> float | None:
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


def evaluate(current: RunResult, baseline: RunResult | None, t: Thresholds) -> Verdict:
    v = Verdict(passed=True)
    base_m = baseline.metrics if baseline else {}
    base_g = baseline.guardrails if baseline else {}

    for name in sorted(set(current.metrics) | set(base_m)):
        d = MetricDelta(name, current.metrics.get(name), base_m.get(name))
        v.metrics.append(d)
        floor = t.min.get(name)
        if floor is not None and d.current is not None and d.current < floor:
            v.failures.append(f"{name} is {d.current:.3f}, below the minimum {floor:.3f}")
        allowed = t.max_drop.get(name, t.default_max_drop)
        if allowed is not None and d.delta is not None and -d.delta > allowed:
            v.failures.append(
                f"{name} dropped {-d.delta:.3f} (from {d.baseline:.3f} to {d.current:.3f}); allowed {allowed:.3f}"
            )
        if d.current is None and d.baseline is not None:
            v.warnings.append(f"{name} was measured in the baseline but not in this run")

    for name in t.min:
        if name not in current.metrics:
            v.warnings.append(f"A minimum is set for {name}, but no case measured it")

    for name in sorted(set(current.guardrails) | set(base_g)):
        d = MetricDelta(name, current.guardrails.get(name), base_g.get(name))
        v.guardrails.append(d)
        if t.guardrails_must_pass and d.current is not None and d.current < 1.0:
            failed = sum(1 for r in current.results if r.guardrails.get(name) is False)
            v.failures.append(f"Guardrail '{name}' failed on {failed} case(s)")

    if current.errors:
        v.failures.append(f"{current.errors} case(s) errored (the pipeline or the judge failed)")

    if baseline:
        if baseline.config_hash and baseline.config_hash != current.config_hash:
            v.warnings.append(
                "Scoring settings (judge, metrics or k) differ from the baseline; deltas may not be like-for-like"
            )
        _case_diffs(current, baseline, v)

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
