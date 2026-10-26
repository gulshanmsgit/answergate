"""Break six industry RAG pipelines on purpose and check that RAGCheck catches every break.

For each industry it:
  1. runs the healthy pipeline and saves it as the baseline (must pass),
  2. runs the healthy pipeline again (must still pass: no false alarm),
  3. switches in each fault from demo_rag.py and runs again (must fail, for the right reason).

It also breaks the healthy runs down by edge case (typos, paraphrases, near misses, ...), so you can see
which kinds of question a pipeline handles badly even when the gate passes.

    python examples/industries/run_demo.py                      # all industries, no judge
    python examples/industries/run_demo.py --industry banking --fault pii_leak
    python examples/industries/run_demo.py --judge ollama       # adds judged metrics (needs `ollama serve`)
    python examples/industries/run_demo.py --judge anthropic --model claude-haiku-5-5

Reports land in examples/industries/out/<industry>/<scenario>.html, with a scorecard in out/scorecard.md.
Exit code 0 means every catchable fault was caught and no healthy run raised a false alarm.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from demo_rag import FAULTS, INDUSTRIES  # noqa: E402

from ragcheck.baseline import evaluate, load_run, save_baseline, save_run  # noqa: E402
from ragcheck.config import Config, load_config, load_golden  # noqa: E402
from ragcheck.guardrails import looks_like_refusal  # noqa: E402
from ragcheck.judges import make_judge  # noqa: E402
from ragcheck.models import RunResult  # noqa: E402
from ragcheck.report import write_reports  # noqa: E402
from ragcheck.runner import run  # noqa: E402
from ragcheck.targets import make_target  # noqa: E402

# What each fault should trip. "judge" faults are invisible to retrieval metrics and regex guardrails.
EXPECTED = {
    "stale_index": "hit_rate / context_recall",
    "top_k_1": "context_recall",
    "rerank_bug": "mrr / ndcg",
    "no_refusal": "refusal guardrail",
    "pii_leak": "pii guardrail",
    "injectable": "injection guardrail",
    "hallucinate": "faithfulness / correctness (judge)",
}
NEEDS_JUDGE = {"hallucinate"}


@dataclass
class Row:
    industry: str
    scenario: str
    expected: str
    passed: bool
    outcome: str
    why: str
    result: RunResult | None = None


def configure(industry: str, judge: str | None, model: str | None, out: Path) -> Config:
    cfg = load_config(HERE / f"{industry}.yaml")
    cfg.workdir = str(out / industry / ".ragcheck")
    if judge:
        cfg.judge.provider = judge  # type: ignore[assignment]
        cfg.judge.model = model
    return cfg


async def one_run(cfg: Config, label: str, fault: str) -> RunResult:
    os.environ["DEMO_FAULT"] = fault
    cases = load_golden(cfg.path(cfg.golden_set))
    target, judge = make_target(cfg), make_judge(cfg)
    try:
        if judge:
            await judge.preflight()
        return await run(cfg, cases, target, judge, label, lambda _: None)
    finally:
        await target.aclose()
        if judge:
            await judge.aclose()


def check(industry: str, cfg: Config, out: Path, scenario: str, fault: str, judged: bool) -> Row:
    workdir = cfg.path(cfg.workdir)
    base_file = workdir / "baseline.json"
    base = load_run(base_file) if base_file.exists() else None
    result = asyncio.run(one_run(cfg, scenario, fault))
    save_run(result, workdir)
    verdict = evaluate(result, base, cfg.thresholds)
    folder = out / industry
    write_reports(result, verdict, base, folder / f"{scenario}.html", folder / f"{scenario}.md", 500)
    why = "; ".join(verdict.failures)[:160]

    if scenario == "baseline":
        # Like `ragcheck baseline`: guardrail failures of the healthy pipeline are accepted as known
        # failures (guardrail_gate: new then warns on them), but nothing else may fail
        known = verdict.failures and all(f.startswith("Guardrail") for f in verdict.failures)
        if verdict.passed or known:
            save_baseline(result, base_file)
        outcome = "PASS" if verdict.passed else "ACCEPTED" if known else "FAIL"
        why = f"known failures accepted: {why}" if known else why
        return Row(industry, scenario, "pass", verdict.passed or bool(known), outcome, why, result)
    if scenario == "healthy-rerun":
        outcome = "PASS" if verdict.passed else "FALSE ALARM"
        return Row(industry, scenario, "pass", verdict.passed, outcome, why, result)
    if not verdict.passed:
        return Row(industry, scenario, EXPECTED[fault], True, "CAUGHT", verdict.failures[0][:160], result)
    if fault in NEEDS_JUDGE and not judged:
        unchanged = "retrieval and guardrails unchanged"
        return Row(industry, scenario, EXPECTED[fault], True, "NEEDS JUDGE", unchanged, result)
    return Row(industry, scenario, EXPECTED[fault], False, "MISSED", "the gate passed", result)


def slices(results: list[RunResult]) -> list[str]:
    """How the healthy pipelines handle each kind of edge case, across all industries."""
    stats: dict[str, dict[str, list[float]]] = {}
    for run_ in results:
        for r in run_.results:
            for tag in (t for t in r.case.tags if t.startswith("edge:")):
                s = stats.setdefault(tag.removeprefix("edge:"), {})
                if r.case.expect_refusal:
                    s.setdefault("declined", []).append(float(r.guardrails.get("refusal", False)))
                else:
                    s.setdefault("hit_rate", []).append(r.scores.get("hit_rate", 0.0))
                    s.setdefault("mrr", []).append(r.scores.get("mrr", 0.0))
                    s.setdefault("answered", []).append(float(not looks_like_refusal(r.answer)))
                for g in ("pii", "injection"):
                    if g in r.guardrails:
                        s.setdefault(g, []).append(float(r.guardrails[g]))

    def mean(values: list[float] | None) -> str:
        return f"{sum(values) / len(values):.2f}" if values else "–"

    lines = [
        "| Edge case | Questions | Hit rate | MRR | Answered | Declined when it should | No PII leaked | Resisted injection |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in stats.items():
        n = len(s.get("hit_rate") or s.get("declined") or [])
        cols = [mean(s.get(k)) for k in ("hit_rate", "mrr", "answered", "declined", "pii", "injection")]
        lines.append(f"| {name} | {n} | " + " | ".join(cols) + " |")
    return lines


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--industry", action="append", choices=INDUSTRIES, help="repeatable; default: all six")
    p.add_argument("--fault", action="append", choices=FAULTS, help="repeatable; default: all seven")
    p.add_argument("--judge", choices=["ollama", "anthropic", "openai"], help="switch on judged metrics")
    p.add_argument("--model", help="judge model (default: the provider's default)")
    p.add_argument("--out", type=Path, default=HERE / "out")
    args = p.parse_args()
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    out: Path = args.out.resolve()
    rows: list[Row] = []
    started = time.perf_counter()
    for industry in args.industry or INDUSTRIES:
        shutil.rmtree(out / industry, ignore_errors=True)  # a fresh baseline every time
        cfg = configure(industry, args.judge, args.model, out)
        plan = [("baseline", ""), ("healthy-rerun", ""), *[(f, f) for f in args.fault or FAULTS]]
        for scenario, fault in plan:
            row = check(industry, cfg, out, scenario, fault, bool(args.judge))
            rows.append(row)
            mark = "✓" if row.passed else "✕"
            print(f"{mark} {industry:<14} {scenario:<14} {row.outcome:<12} {row.why}")
    os.environ.pop("DEMO_FAULT", None)

    faults = [r for r in rows if r.scenario not in ("baseline", "healthy-rerun")]
    caught = sum(r.outcome == "CAUGHT" for r in faults)
    needs_judge = sum(r.outcome == "NEEDS JUDGE" for r in faults)
    problems = [r for r in rows if not r.passed]
    baselines = [r.result for r in rows if r.scenario == "baseline" and r.result]
    questions = sum(len(b.results) for b in baselines)
    checks = sum(len(r.result.results) for r in rows if r.result)
    summary = (
        f"{questions} questions · {checks} question checks · {caught} of {len(faults)} injected faults caught · "
        f"{needs_judge} need a judge · {sum(r.outcome == 'FALSE ALARM' for r in rows)} false alarms · "
        f"{len(problems)} problems · judge: {args.judge or 'none'} · {time.perf_counter() - started:.1f} s"
    )
    edge = slices(baselines)
    lines = [
        "# RAGCheck industry demo scorecard",
        "",
        summary,
        "",
        "## Healthy pipelines by edge case",
        "",
        *edge,
        "",
        "## Runs",
        "",
        "| Industry | Scenario | Should trip | Outcome | First reason |",
        "|---|---|---|---|---|",
        *[
            f"| {r.industry} | {r.scenario} | {r.expected} | {r.outcome} | {r.why.replace('|', '/')} |"
            for r in rows
        ],
    ]
    out.mkdir(parents=True, exist_ok=True)
    (out / "scorecard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print("\n".join(edge))
    print()
    print(summary)
    print(f"scorecard: {out / 'scorecard.md'}")
    print(f"reports:   {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
