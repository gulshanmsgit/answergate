"""Reports: a Markdown summary for the PR comment and a self-contained HTML page."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from answergate.baseline import Verdict
from answergate.metrics.retrieval import is_expected
from answergate.models import RunResult

LABELS = {
    "hit_rate": "Hit rate",
    "mrr": "MRR",
    "ndcg": "nDCG",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "faithfulness": "Faithfulness",
    "relevance": "Answer relevance",
    "correctness": "Correctness",
    "consistency": "Answer consistency",
    "pii": "No PII leaked",
    "injection": "Resists injection",
    "refusal": "Refuses off-topic",
}

# Marker so the GitHub Action can find and update its own PR comment
COMMENT_MARKER = "<!-- answergate-report -->"


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " ").capitalize())


def _fmt_delta(delta: float | None) -> str:
    if delta is None:
        return "—"
    if abs(delta) < 0.0005:
        return "±0"
    return f"{'+' if delta > 0 else '−'}{abs(delta):.3f}"


def _code(text: str) -> str:
    """Inline code that survives a Markdown table: escape pipes, and fence backticks with a longer run."""
    text = text.replace("|", "\\|").replace("\n", " ")
    fence = "``" if "`" in text else "`"
    pad = " " if "`" in text else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _cell(text: str, limit: int) -> str:
    """Plain text that fits one Markdown table cell."""
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _drop(before: str, after: str) -> float:
    """How much a regressed score fell; 1.0 for a non-numeric change such as a guardrail that broke."""
    try:
        return float(before) - float(after)
    except ValueError:
        return 1.0


def headline(run: RunResult, verdict: Verdict) -> str:
    """One line a reviewer reads first, e.g. "Correctness dropped 6% (0.925 → 0.869) · 12 of 44 questions got worse"."""
    drops = [
        f"{label(d.name)} dropped {-d.delta / d.baseline:.0%} ({d.baseline:.3f} → {d.current:.3f})"
        for d in verdict.metrics
        if d.delta is not None
        and d.delta < 0
        and d.baseline
        and d.current is not None
        and any(f.startswith(f"{d.name} ") for f in verdict.failures)
    ]
    worse = len(verdict.regressed_cases)
    if worse:
        drops.append(f"{worse} of {len(run.results)} questions got worse")
    return " · ".join(drops)


def markdown(run: RunResult, verdict: Verdict, report_url: str | None = None) -> str:
    status = "✅ **AnswerGate passed**" if verdict.passed else "❌ **AnswerGate failed**"
    lines = [COMMENT_MARKER, f"### {status}", ""]
    if summary := headline(run, verdict):
        lines += [f"**{summary}**", ""]
    if verdict.pipeline_changes:
        shown = verdict.pipeline_changes[:10]
        changed = " · ".join(
            f"{_code(c.key)} {_cell(c.before, 60) or 'not reported'} → {_cell(c.after, 60) or 'not reported'}"
            for c in shown
        )
        more = len(verdict.pipeline_changes) - len(shown)
        lines += [f"**What changed in the pipeline:** {changed}" + (f" · and {more} more" if more else ""), ""]
    lines.append(
        f"{len(run.results)} questions · p50 {run.latency_p50_ms:.0f} ms · p95 {run.latency_p95_ms:.0f} ms"
        + (f" · {run.errors} errors" if run.errors else "")
    )
    if verdict.failures:
        lines += ["", "**Why it failed**", *[f"- {f}" for f in verdict.failures]]
    if verdict.regressions:
        answers = {r.case.id: r.answer for r in run.results}
        by_case: dict[str, list[str]] = {}
        questions: dict[str, str] = {}
        lost: dict[str, float] = {}
        for r in verdict.regressions:
            by_case.setdefault(r.case_id, []).append(f"{label(r.what)} {r.before} → {r.after}")
            questions[r.case_id] = r.question
            lost[r.case_id] = lost.get(r.case_id, 0.0) + _drop(r.before, r.after)
        # Worst first: the questions that lost the most across their metrics
        shown = sorted(by_case, key=lambda cid: -lost[cid])[:10]
        lines += ["", f"**Questions that got worse** ({len(shown)} of {len(by_case)})", ""]
        lines += ["| Question | What got worse | Answer now |", "|---|---|---|"]
        for cid in shown:
            lines.append(
                f"| {_cell(questions[cid], 90)} | {'<br>'.join(by_case[cid])} | {_cell(answers.get(cid, ''), 140)} |"
            )
    lines += ["", "| Metric | Baseline | This run | Change [95% CI] |", "|---|---:|---:|---:|"]
    for d in verdict.metrics + verdict.guardrails:
        base = f"{d.baseline:.3f}" if d.baseline is not None else "—"
        cur = f"{d.current:.3f}" if d.current is not None else "—"
        ci = f" [{_fmt_delta(d.ci_low)}, {_fmt_delta(d.ci_high)}]" if d.ci_low is not None else ""
        lines.append(f"| {label(d.name)} | {base} | {cur} | {_fmt_delta(d.delta)}{ci} |")
    if verdict.regressions:
        lines += ["", f"<details><summary>Every change ({len(verdict.regressions)})</summary>", ""]
        lines += ["| Case | Metric | Before | After |", "|---|---|---:|---:|"]
        for r in verdict.regressions[:50]:
            lines.append(f"| {_code(r.case_id)} | {label(r.what)} | {r.before} | {r.after} |")
        lines += ["", "</details>"]
    if verdict.warnings:
        lines += ["", *[f"> ⚠️ {w}" for w in verdict.warnings]]
    if report_url:
        lines += ["", f"[Full report]({report_url})"]
    return "\n".join(lines) + "\n"


def html(run: RunResult, verdict: Verdict, baseline: RunResult | None, limit: int = 500) -> str:
    env = Environment(
        loader=PackageLoader("answergate", "templates"),
        # Answers and contexts are untrusted text; the template's .j2 suffix would disable the default
        autoescape=select_autoescape(enabled_extensions=("html", "j2"), default=True),
    )
    env.filters["label"] = label
    env.filters["delta"] = _fmt_delta
    env.globals["is_expected"] = lambda cid, expected: is_expected(cid, expected, run.source_match)
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
    # A 10,000-question report would be unusable: show every problem, then passing questions up to the limit
    problems = [
        r
        for r in ordered
        if r.case.id in regressed or r.error or r.judge_error or not all(r.guardrails.values())
    ]
    shown = problems + [r for r in ordered if r not in problems][: max(0, limit - len(problems))]
    hidden = len(ordered) - len(shown)
    return env.get_template("report.html.j2").render(
        run=run,
        verdict=verdict,
        baseline=baseline,
        before=before,
        regressed=regressed,
        results=shown,
        hidden=hidden,
    )


def write_reports(
    run: RunResult,
    verdict: Verdict,
    baseline: RunResult | None,
    html_path: Path | None,
    md_path: Path | None,
    limit: int = 500,
) -> None:
    if html_path:
        html_path.write_text(html(run, verdict, baseline, limit), encoding="utf-8")
    if md_path:
        md_path.write_text(markdown(run, verdict), encoding="utf-8")
