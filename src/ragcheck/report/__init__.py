"""Reports: a Markdown summary for the PR comment and a self-contained HTML page."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from ragcheck.baseline import Verdict
from ragcheck.models import RunResult

LABELS = {
    "hit_rate": "Hit rate",
    "mrr": "MRR",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "faithfulness": "Faithfulness",
    "relevance": "Answer relevance",
    "correctness": "Correctness",
    "pii": "No PII leaked",
    "injection": "Resists injection",
    "refusal": "Refuses off-topic",
}

# Marker so the GitHub Action can find and update its own PR comment
COMMENT_MARKER = "<!-- ragcheck-report -->"


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " ").capitalize())


def _fmt_delta(delta: float | None) -> str:
    if delta is None:
        return "—"
    if abs(delta) < 0.0005:
        return "±0"
    return f"{'+' if delta > 0 else '−'}{abs(delta):.3f}"


def markdown(run: RunResult, verdict: Verdict, report_url: str | None = None) -> str:
    status = "✅ **RAGCheck passed**" if verdict.passed else "❌ **RAGCheck failed**"
    lines = [COMMENT_MARKER, f"### {status}", ""]
    lines.append(
        f"{len(run.results)} questions · p50 {run.latency_p50_ms:.0f} ms · p95 {run.latency_p95_ms:.0f} ms"
        + (f" · {run.errors} errors" if run.errors else "")
    )
    lines += ["", "| Metric | Baseline | This run | Change |", "|---|---:|---:|---:|"]
    for d in verdict.metrics + verdict.guardrails:
        base = f"{d.baseline:.3f}" if d.baseline is not None else "—"
        cur = f"{d.current:.3f}" if d.current is not None else "—"
        lines.append(f"| {label(d.name)} | {base} | {cur} | {_fmt_delta(d.delta)} |")
    if verdict.failures:
        lines += ["", "**Why it failed**", *[f"- {f}" for f in verdict.failures]]
    if verdict.regressions:
        lines += ["", f"<details><summary>{len(verdict.regressed_cases)} question(s) regressed</summary>", ""]
        lines += ["| Case | Metric | Before | After |", "|---|---|---:|---:|"]
        for r in verdict.regressions[:25]:
            lines.append(f"| `{r.case_id}` | {label(r.what)} | {r.before} | {r.after} |")
        lines += ["", "</details>"]
    if verdict.warnings:
        lines += ["", *[f"> ⚠️ {w}" for w in verdict.warnings]]
    if report_url:
        lines += ["", f"[Full report]({report_url})"]
    return "\n".join(lines) + "\n"


def html(run: RunResult, verdict: Verdict, baseline: RunResult | None) -> str:
    env = Environment(
        loader=PackageLoader("ragcheck", "templates"),
        # Answers and contexts are untrusted text; the template's .j2 suffix would disable the default
        autoescape=select_autoescape(enabled_extensions=("html", "j2"), default=True),
    )
    env.filters["label"] = label
    env.filters["delta"] = _fmt_delta
    before = {r.case.id: r for r in baseline.results} if baseline else {}
    regressed = {r.case_id for r in verdict.regressions}
    # Regressed and failing questions first, so the reader sees problems without scrolling
    ordered = sorted(
        run.results,
        key=lambda r: (
            r.case.id not in regressed,
            not (r.error or not all(r.guardrails.values())),
            r.case.id,
        ),
    )
    return env.get_template("report.html.j2").render(
        run=run, verdict=verdict, baseline=baseline, before=before, regressed=regressed, results=ordered
    )


def write_reports(
    run: RunResult, verdict: Verdict, baseline: RunResult | None, html_path: Path | None, md_path: Path | None
) -> None:
    if html_path:
        html_path.write_text(html(run, verdict, baseline), encoding="utf-8")
    if md_path:
        md_path.write_text(markdown(run, verdict), encoding="utf-8")
