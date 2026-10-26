"""Break six industry RAG pipelines on purpose and check that RAGCheck catches every break.

For each industry it:
  1. runs the healthy pipeline and saves it as the baseline (must pass),
  2. runs the healthy pipeline again (must still pass: no false alarm),
  3. switches in each fault from demo_rag.py and runs again (must fail, for the right reason).

    python examples/industries/run_demo.py                      # all industries, no judge, ~10 s
    python examples/industries/run_demo.py --industry banking --fault pii_leak
    python examples/industries/run_demo.py --judge ollama       # adds judged metrics (needs `ollama serve`)
    python examples/industries/run_demo.py --judge anthropic --model claude-haiku-5-5

Reports land in examples/industries/out/<industry>/<fault>.html, with a scorecard in out/scorecard.md.
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

from ragcheck.baseline import evaluate, save_baseline, save_run  # noqa: E402
from ragcheck.config import Config, load_config, load_golden  # noqa: E402
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
    base = (
        RunResult.model_validate_json(base_file.read_text(encoding="utf-8")) if base_file.exists() else None
    )
    result = asyncio.run(one_run(cfg, scenario, fault))
    save_run(result, workdir)
    verdict = evaluate(result, base, cfg.thresholds)
    folder = out / industry
    write_reports(result, verdict, base, folder / f"{scenario}.html", folder / f"{scenario}.md", 500)

    if scenario == "baseline":
        if verdict.passed:
            save_baseline(result, base_file)
        ok, outcome = verdict.passed, "PASS" if verdict.passed else "FAIL"
        return Row(industry, scenario, "pass", ok, outcome, "; ".join(verdict.failures)[:160])
    if scenario == "healthy-rerun":
        ok = verdict.passed
        return Row(
            industry, scenario, "pass", ok, "PASS" if ok else "FALSE ALARM", "; ".join(verdict.failures)[:160]
        )

    caught = not verdict.passed
    if caught:
        return Row(industry, scenario, EXPECTED[fault], True, "CAUGHT", verdict.failures[0][:160])
    if fault in NEEDS_JUDGE and not judged:
        return Row(
            industry, scenario, EXPECTED[fault], True, "NEEDS JUDGE", "retrieval and guardrails unchanged"
        )
    return Row(industry, scenario, EXPECTED[fault], False, "MISSED", "the gate passed")


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
    summary = (
        f"{caught} of {len(faults)} injected faults caught · {needs_judge} need a judge · "
        f"{sum(r.outcome == 'FALSE ALARM' for r in rows)} false alarms · {len(problems)} problems · "
        f"judge: {args.judge or 'none'} · {time.perf_counter() - started:.1f} s"
    )
    lines = [
        "# RAGCheck industry demo scorecard",
        "",
        summary,
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
    print(f"\n{summary}\nscorecard: {out / 'scorecard.md'}\nreports:   {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
