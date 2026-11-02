"""The `answergate` command."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import os
import shutil
import sys
import traceback
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, NoReturn, ParamSpec, TypeVar

import typer

from answergate import __version__
from answergate.baseline import Verdict, evaluate, latest_run, load_run, save_baseline, save_run
from answergate.config import Config, ConfigError, load_config, load_golden
from answergate.judges import FatalJudgeError, JudgeError, make_judge
from answergate.models import CaseResult, RunResult
from answergate.report import label, write_reports
from answergate.runner import build_run
from answergate.runner import run as run_pipeline
from answergate.selection import select
from answergate.targets import make_target

app = typer.Typer(
    help="Regression tests for RAG pipelines: catch quality drops in CI before your users do.",
    no_args_is_help=True,
    add_completion=False,
)

# Exit codes: 0 passed, 1 quality gate failed, 2 could not run
EXIT_FAILED, EXIT_ERROR = 1, 2


@app.callback()
def _setup() -> None:
    # Windows consoles default to a legacy code page that can't print the report's symbols
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to answergate.yaml")]


def _die(msg: str) -> NoReturn:
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(EXIT_ERROR)


P = ParamSpec("P")
R = TypeVar("R")


def _guard(fn: Callable[P, R]) -> Callable[P, R]:
    """Turn an unexpected crash into exit code 2, so CI never mistakes a AnswerGate bug for a regression."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            # Exit codes, aborts and usage errors are the CLI framework's own control flow
            if type(e).__module__.startswith(("typer", "click")):
                raise
            typer.secho(f"AnswerGate crashed: {type(e).__name__}: {e}", fg=typer.colors.RED, err=True)
            if os.environ.get("ANSWERGATE_DEBUG"):
                traceback.print_exc()
            else:
                typer.echo("Set ANSWERGATE_DEBUG=1 to see the traceback.", err=True)
            raise typer.Exit(EXIT_ERROR) from e

    return wrapper


def _read_run(path: Path, what: str) -> RunResult:
    try:
        return load_run(path)
    except (OSError, ValueError) as e:
        reason = str(e).splitlines()[0] if str(e) else type(e).__name__
        _die(
            f"Cannot read the {what} at {path}: {reason}\n"
            "Delete it, or set a new one with `answergate baseline`."
        )


def _load(config: Path) -> Config:
    try:
        return load_config(config)
    except ConfigError as e:
        _die(str(e))


def _baseline_path(cfg: Config) -> Path:
    return cfg.path(cfg.workdir) / "baseline.json"


def _print_verdict(run: RunResult, verdict: Verdict) -> None:
    typer.echo("")
    if verdict.pipeline_changes:
        typer.secho("  Pipeline changes since the baseline:", bold=True)
        for c in verdict.pipeline_changes:
            typer.echo(f"    {c.key}: {c.before or '(not reported)'} → {c.after or '(not reported)'}")
        typer.echo("")
    for d in verdict.metrics + verdict.guardrails:
        cur = f"{d.current:.3f}" if d.current is not None else "  —  "
        delta = ""
        if d.delta is not None:
            colour = (
                typer.colors.GREEN if d.delta > 0.0005 else typer.colors.RED if d.delta < -0.0005 else None
            )
            delta = typer.style(f"{d.delta:+.3f}", fg=colour)
            if d.ci_low is not None and d.ci_high is not None:
                delta += typer.style(f"  [{d.ci_low:+.3f}, {d.ci_high:+.3f}]", dim=True)
        typer.echo(f"  {label(d.name):<20} {cur}  {delta}")
    typer.echo(
        f"\n  {len(run.results)} questions · p50 {run.latency_p50_ms:.0f} ms · p95 {run.latency_p95_ms:.0f} ms"
    )
    if verdict.regressions:
        typer.echo(f"  {len(verdict.regressed_cases)} question(s) regressed:")
        for r in verdict.regressions[:10]:
            typer.echo(f"    - {r.case_id}: {label(r.what)} {r.before} → {r.after}")
    for w in verdict.warnings:
        typer.secho(f"  ! {w}", fg=typer.colors.YELLOW)
    typer.echo("")
    if verdict.passed:
        typer.secho("PASSED", fg=typer.colors.GREEN, bold=True)
    else:
        typer.secho("FAILED", fg=typer.colors.RED, bold=True)
        for f in verdict.failures:
            typer.echo(f"  ✕ {f}")


@app.command()
@_guard
def init(
    directory: Annotated[Path, typer.Argument(help="Where to create the files")] = Path("."),
    force: Annotated[bool, typer.Option(help="Overwrite existing files")] = False,
) -> None:
    """Create a starter answergate.yaml, golden.jsonl and an injection test document."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "answergate.yaml": "answergate.yaml",
        "golden.jsonl": "golden.jsonl",
        "canary-doc.md": "canary-doc.md",
    }
    for src, dest in files.items():
        out = directory / dest
        if out.exists() and not force:
            typer.echo(f"skip   {out} (exists)")
            continue
        with resources.as_file(resources.files("answergate") / "templates" / src) as p:
            shutil.copyfile(p, out)
        typer.echo(f"create {out}")
    typer.echo(
        "\nNext: point target.url at your pipeline, write real questions in golden.jsonl, then run `answergate run`."
    )


@app.command()
@_guard
def run(
    config: ConfigOpt = Path("answergate.yaml"),
    label_: Annotated[
        str, typer.Option("--label", "-l", help="A name for this run, e.g. a branch or commit")
    ] = "",
    baseline: Annotated[Path | None, typer.Option(help="Baseline run to compare with")] = None,
    html: Annotated[Path | None, typer.Option(help="Write the HTML report here")] = Path(
        "answergate-report.html"
    ),
    md: Annotated[Path | None, typer.Option(help="Write a Markdown summary here (for PR comments)")] = None,
    save_as_baseline: Annotated[
        bool, typer.Option("--save-baseline", help="Make this run the new baseline")
    ] = False,
    no_judge: Annotated[bool, typer.Option("--no-judge", help="Skip LLM-judged metrics")] = False,
    shard: Annotated[
        str | None,
        typer.Option(help="Run one part of the golden set, e.g. 2/4; combine with `answergate merge`"),
    ] = None,
    sample: Annotated[int | None, typer.Option(help="Ask a random subset of this many questions")] = None,
    seed: Annotated[int, typer.Option(help="Seed for --sample (same seed, same questions)")] = 0,
    tag: Annotated[list[str] | None, typer.Option(help="Only questions with this tag (repeatable)")] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Continue an interrupted run, skipping finished questions")
    ] = False,
    report_limit: Annotated[
        int, typer.Option(help="Most questions shown in the HTML report (problems always shown)")
    ] = 500,
) -> None:
    """Ask every golden question, score the answers, and compare with the baseline."""
    cfg = _load(config)
    if no_judge:
        cfg.judge.provider = "none"
    try:
        all_cases = load_golden(cfg.path(cfg.golden_set))
        cases = select(all_cases, shard, sample, seed, tag)
        target = make_target(cfg)
        judge = make_judge(cfg)
    except (ConfigError, JudgeError, FatalJudgeError) as e:
        _die(str(e))
    subset = len(cases) < len(all_cases)
    if subset and save_as_baseline:
        _die("--save-baseline needs the whole golden set: a partial baseline would skew every comparison.")
    if subset:
        parts = [f"shard {shard}" if shard else "", f"sample {len(cases)} (seed {seed})" if sample else ""]
        parts.append(f"tags {','.join(tag)}" if tag else "")
        label_ = " · ".join(p for p in [label_, *parts] if p)
    # One checkpoint per selection, so parallel shards in one workdir never share a file
    ids = "\n".join(c.id for c in cases)
    checkpoint = cfg.path(cfg.workdir) / f"inprogress-{hashlib.sha256(ids.encode()).hexdigest()[:10]}.jsonl"

    done = 0

    def tick(r: CaseResult) -> None:
        nonlocal done
        done += 1
        if not sys.stdout.isatty():
            # Keep CI logs clean: only failures, one per line
            if r.error:
                typer.secho(f"  ✕ {r.case.id}: {r.error}", fg=typer.colors.RED)
            return
        mark = typer.style("✕", fg=typer.colors.RED) if r.error else typer.style("·", fg=typer.colors.GREEN)
        typer.echo(f"\r  {mark} {done}/{len(cases)} {r.case.id[:40]:<40}", nl=False)

    async def go() -> RunResult:
        try:
            if judge:
                await judge.preflight()
            return await run_pipeline(cfg, cases, target, judge, label_, tick, checkpoint, resume)
        finally:
            await target.aclose()
            if judge:
                await judge.aclose()

    base_path = baseline or _baseline_path(cfg)
    # Read the baseline before calling the pipeline, so a bad file fails fast
    base = _read_run(base_path, "baseline") if base_path.exists() else None

    of = f" of {len(all_cases)}" if subset else ""
    typer.echo(
        f"AnswerGate {__version__} · {len(cases)}{of} questions · judge: {judge.name if judge else 'none'}"
    )
    try:
        result = asyncio.run(go())
    except FatalJudgeError as e:
        typer.echo("")
        _die(f"Judge setup problem: {e}")
    except KeyboardInterrupt:
        typer.echo("")
        _die(f"Interrupted after {done} questions. They are saved: continue with `answergate run --resume`.")
    typer.echo("")
    checkpoint.unlink(missing_ok=True)  # the full run is saved below

    workdir = cfg.path(cfg.workdir)
    saved = save_run(result, workdir)
    if base is None:
        typer.secho("  No baseline yet: only absolute minimums are checked.", fg=typer.colors.YELLOW)

    verdict = evaluate(result, base, cfg.thresholds)
    _print_verdict(result, verdict)
    write_reports(result, verdict, base, html, md, report_limit)
    typer.echo(f"\n  run saved to {saved}")
    if html:
        typer.echo(f"  report: {html.resolve()}")
    if judge:
        typer.echo(f"  judge cache: {judge.hits} hits, {judge.misses} new calls")

    if save_as_baseline:
        if verdict.passed:
            save_baseline(result, _baseline_path(cfg))
            typer.echo(f"  saved as baseline: {_baseline_path(cfg)}")
        else:
            typer.secho(
                "  not saved as baseline, because the run failed. "
                "To accept it anyway, run `answergate baseline`.",
                fg=typer.colors.YELLOW,
            )

    raise typer.Exit(0 if verdict.passed else EXIT_FAILED)


@app.command()
@_guard
def compare(
    current: Annotated[Path | None, typer.Argument(help="Run file (default: the latest run)")] = None,
    baseline: Annotated[Path | None, typer.Option(help="Baseline run file")] = None,
    config: ConfigOpt = Path("answergate.yaml"),
    html: Annotated[Path | None, typer.Option(help="Write the HTML report here")] = None,
    md: Annotated[Path | None, typer.Option(help="Write a Markdown summary here")] = None,
) -> None:
    """Re-check a saved run against a baseline without calling the pipeline again."""
    cfg = _load(config)
    workdir = cfg.path(cfg.workdir)
    cur_path = current or latest_run(workdir)
    if cur_path is None:
        _die("No runs yet. Run `answergate run` first.")
    base_path = baseline or _baseline_path(cfg)
    if not base_path.exists():
        _die(f"Baseline not found: {base_path}. Save one with `answergate baseline`.")
    cur, base = _read_run(cur_path, "run"), _read_run(base_path, "baseline")
    verdict = evaluate(cur, base, cfg.thresholds)
    _print_verdict(cur, verdict)
    write_reports(cur, verdict, base, html, md)
    raise typer.Exit(0 if verdict.passed else EXIT_FAILED)


@app.command()
@_guard
def merge(
    runs: Annotated[list[Path], typer.Argument(help="Run files from each shard")],
    config: ConfigOpt = Path("answergate.yaml"),
    baseline: Annotated[Path | None, typer.Option(help="Baseline run file")] = None,
    html: Annotated[Path | None, typer.Option(help="Write the HTML report here")] = Path(
        "answergate-report.html"
    ),
    md: Annotated[Path | None, typer.Option(help="Write a Markdown summary here")] = None,
    save_as_baseline: Annotated[
        bool, typer.Option("--save-baseline", help="Make the merged run the new baseline if it passes")
    ] = False,
    report_limit: Annotated[int, typer.Option(help="Most questions shown in the HTML report")] = 500,
) -> None:
    """Combine shard runs (`run --shard i/n`) into one run, then gate it like a normal run."""
    cfg = _load(config)
    parts = [_read_run(p, "run") for p in runs]
    if len({p.config_hash for p in parts}) > 1:
        _die(
            "These runs used different scoring settings (config fingerprints differ): merge runs from one config."
        )
    seen: dict[str, str] = {}
    results = []
    for path, part in zip(runs, parts, strict=True):
        for r in part.results:
            if r.case.id in seen:
                _die(
                    f"Question '{r.case.id}' is in both {seen[r.case.id]} and {path.name}: shards must not overlap."
                )
            seen[r.case.id] = path.name
            results.append(r)
    label_ = "merged: " + " + ".join(p.label or p.id for p in parts)
    merged = build_run(results, parts[0].config_hash, label_, parts[0].source_match)
    saved = save_run(merged, cfg.path(cfg.workdir))
    base_path = baseline or _baseline_path(cfg)
    base = _read_run(base_path, "baseline") if base_path.exists() else None
    total = len(load_golden(cfg.path(cfg.golden_set)))
    verdict = evaluate(merged, base, cfg.thresholds)
    if len(results) < total:
        verdict.warnings.append(f"The merged shards cover {len(results)} of {total} golden questions")
    typer.echo(f"Merged {len(parts)} runs · {len(results)} questions")
    _print_verdict(merged, verdict)
    write_reports(merged, verdict, base, html, md, report_limit)
    typer.echo(f"\n  run saved to {saved}")
    if save_as_baseline:
        if verdict.passed and len(results) == total:
            save_baseline(merged, _baseline_path(cfg))
            typer.echo(f"  saved as baseline: {_baseline_path(cfg)}")
        else:
            typer.secho(
                "  not saved as baseline: the merged run failed or is missing questions",
                fg=typer.colors.YELLOW,
            )
    raise typer.Exit(0 if verdict.passed else EXIT_FAILED)


@app.command()
@_guard
def calibrate(
    config: ConfigOpt = Path("answergate.yaml"),
    labels: Annotated[
        Path | None, typer.Option(help="Your own labelled cases (JSONL). Default: the built-in set")
    ] = None,
    min_accuracy: Annotated[float, typer.Option(help="Fail if any metric's accuracy is below this")] = 0.85,
) -> None:
    """Check the judge against labelled cases: how often does it agree with the right answer?"""
    from answergate.calibrate import calibrate as run_calibration
    from answergate.calibrate import load_labels

    cfg = _load(config)
    try:
        cases = load_labels(labels)
        judge = make_judge(cfg)
    except (ConfigError, FatalJudgeError) as e:
        _die(str(e))
    if judge is None:
        _die("Calibration needs a judge: set judge.provider in the config.")

    async def go() -> dict[str, Any]:
        try:
            await judge.preflight()
            return await run_calibration(cases, judge, cfg)
        finally:
            await judge.aclose()

    typer.echo(f"Calibrating {judge.name} on {len(cases)} labelled cases (samples: {cfg.judge.samples})\n")
    try:
        results = asyncio.run(go())
    except FatalJudgeError as e:
        _die(f"Judge setup problem: {e}")

    typer.echo(f"  {'metric':<14} {'cases':>5} {'accuracy':>9} {'kappa':>7}  errors")
    worst = 1.0
    for r in results.values():
        worst = min(worst, r.accuracy)
        colour = typer.colors.GREEN if r.accuracy >= min_accuracy else typer.colors.RED
        acc = typer.style(f"{r.accuracy:>9.2f}", fg=colour)
        typer.echo(f"  {r.kind:<14} {r.n:>5} {acc} {r.kappa:>7.2f}  {r.errors}")
    misses = [m for r in results.values() for m in r.misses]
    if misses:
        typer.echo("\n  Disagreements:")
        for m in misses:
            typer.echo(f"    - {m}")
    typer.echo("")
    if worst >= min_accuracy:
        typer.secho(
            f"CALIBRATED: every metric at or above {min_accuracy:.2f} accuracy", fg=typer.colors.GREEN
        )
        raise typer.Exit(0)
    typer.secho(
        f"NOT CALIBRATED: lowest accuracy {worst:.2f} < {min_accuracy:.2f}. "
        "Try a stronger judge model, judge.samples: 3, or check the labels.",
        fg=typer.colors.RED,
    )
    raise typer.Exit(EXIT_FAILED)


@app.command()
@_guard
def generate(
    docs: Annotated[Path, typer.Argument(help="Folder of documents (.md, .mdx, .txt, .rst)")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Golden set to write")] = Path(
        "golden.generated.jsonl"
    ),
    n: Annotated[int, typer.Option("--n", "-n", min=1, help="How many questions to write")] = 40,
    mode: Annotated[
        str,
        typer.Option(
            help="structure: no LLM, from headings and definitions; llm: the judge model writes them"
        ),
    ] = "structure",
    refusals: Annotated[
        int, typer.Option(min=0, help="Off-topic questions to add, which must be declined")
    ] = 3,
    seed: Annotated[int, typer.Option(help="Same seed and documents, same questions")] = 7,
    config: ConfigOpt = Path("answergate.yaml"),
    force: Annotated[bool, typer.Option(help="Overwrite the output file")] = False,
) -> None:
    """Write a golden set from your documents. Each question's expected source is the passage it was written from."""
    from answergate.generate import generate_llm, generate_structure, load_sections, with_refusals

    if mode not in ("structure", "llm"):
        _die(f"--mode must be structure or llm, not {mode!r}")
    if not docs.is_dir():
        _die(f"Not a folder: {docs}")
    if out.exists() and not force:
        _die(f"{out} exists. Review and keep it, or pass --force to overwrite.")
    sections = load_sections(docs)
    if not sections:
        _die(f"No document sections found in {docs} (looked for .md, .mdx, .txt and .rst files).")

    if mode == "structure":
        result = generate_structure(sections, n, seed)
    else:
        cfg = _load(config)
        try:
            judge = make_judge(cfg)
        except FatalJudgeError as e:
            _die(str(e))
        if judge is None:
            _die("--mode llm uses the judge model from the config: set judge.provider (e.g. ollama).")

        async def go() -> Any:
            try:
                await judge.preflight()
                return await generate_llm(sections, n, judge, seed, batch=max(1, cfg.judge.concurrency) * 2)
            finally:
                await judge.aclose()

        typer.echo(f"Writing up to {n} questions with {judge.name}…")
        try:
            result = asyncio.run(go())
        except FatalJudgeError as e:
            _die(f"Judge setup problem: {e}")

    with_refusals(result, refusals)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(c.model_dump_json(exclude_defaults=True) + "\n" for c in result.cases), encoding="utf-8"
    )
    answerable = sum(not c.expect_refusal for c in result.cases)
    typer.echo(
        f"Wrote {len(result.cases)} questions to {out}: {answerable} from "
        f"{len({c.expected_sources[0] for c in result.cases if c.expected_sources})} documents, "
        f"{len(result.cases) - answerable} to decline."
    )
    if result.dropped:
        typer.echo("  Dropped: " + ", ".join(f"{k} {v}" for k, v in result.dropped.most_common()))
    if answerable < n:
        typer.secho(f"  Only {answerable} of {n} questions passed the checks.", fg=typer.colors.YELLOW)
    typer.echo("Review the file like code, delete weak questions, then set golden_set to it.")


@app.command("baseline")
@_guard
def set_baseline(
    run_file: Annotated[Path | None, typer.Argument(help="Run file (default: the latest run)")] = None,
    config: ConfigOpt = Path("answergate.yaml"),
    force: Annotated[bool, typer.Option(help="Accept a run with pipeline or judge errors anyway")] = False,
) -> None:
    """Make a saved run the baseline that future runs are compared with."""
    cfg = _load(config)
    src = run_file or latest_run(cfg.path(cfg.workdir))
    if src is None:
        _die("No runs yet. Run `answergate run` first.")
    run = _read_run(src, "run")
    # A baseline missing answers or scores makes every later comparison meaningless: a question that
    # errored can't regress, and a metric absent from the baseline is never gated
    problems = []
    if run.errors:
        first = next(r.error for r in run.results if r.error)
        problems.append(f"{run.errors} of {len(run.results)} questions got no answer (e.g. {first})")
    if run.judge_errors:
        problems.append(f"{run.judge_errors} questions are missing judge scores")
    if not run.metrics and not run.guardrails:
        problems.append("it has no scores at all")
    if problems and not force:
        _die(
            f"Not using {src.name} as the baseline: {'; '.join(problems)[:400]}. "
            "Fix the cause and run again, or pass --force to accept it anyway."
        )
    save_baseline(run, _baseline_path(cfg))
    typer.echo(f"Baseline set to {src.name} → {_baseline_path(cfg)}")


@app.command()
@_guard
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
