"""The `ragcheck` command."""

from __future__ import annotations

import asyncio
import shutil
import sys
from importlib import resources
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from ragcheck import __version__
from ragcheck.baseline import Verdict, evaluate, latest_run, load_run, save_baseline, save_run
from ragcheck.config import Config, ConfigError, load_config, load_golden
from ragcheck.judges import JudgeError, make_judge
from ragcheck.models import CaseResult, RunResult
from ragcheck.report import label, write_reports
from ragcheck.runner import run as run_pipeline
from ragcheck.targets import make_target

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


ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to ragcheck.yaml")]


def _die(msg: str) -> NoReturn:
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(EXIT_ERROR)


def _load(config: Path) -> Config:
    try:
        return load_config(config)
    except ConfigError as e:
        _die(str(e))


def _baseline_path(cfg: Config) -> Path:
    return cfg.path(cfg.workdir) / "baseline.json"


def _print_verdict(run: RunResult, verdict: Verdict) -> None:
    typer.echo("")
    for d in verdict.metrics + verdict.guardrails:
        cur = f"{d.current:.3f}" if d.current is not None else "  —  "
        delta = ""
        if d.delta is not None:
            colour = (
                typer.colors.GREEN if d.delta > 0.0005 else typer.colors.RED if d.delta < -0.0005 else None
            )
            delta = typer.style(f"{d.delta:+.3f}", fg=colour)
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
def init(
    directory: Annotated[Path, typer.Argument(help="Where to create the files")] = Path("."),
    force: Annotated[bool, typer.Option(help="Overwrite existing files")] = False,
) -> None:
    """Create a starter ragcheck.yaml, golden.jsonl and an injection test document."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "ragcheck.yaml": "ragcheck.yaml",
        "golden.jsonl": "golden.jsonl",
        "canary-doc.md": "canary-doc.md",
    }
    for src, dest in files.items():
        out = directory / dest
        if out.exists() and not force:
            typer.echo(f"skip   {out} (exists)")
            continue
        with resources.as_file(resources.files("ragcheck") / "templates" / src) as p:
            shutil.copyfile(p, out)
        typer.echo(f"create {out}")
    typer.echo(
        "\nNext: point target.url at your pipeline, write real questions in golden.jsonl, then run `ragcheck run`."
    )


@app.command()
def run(
    config: ConfigOpt = Path("ragcheck.yaml"),
    label_: Annotated[
        str, typer.Option("--label", "-l", help="A name for this run, e.g. a branch or commit")
    ] = "",
    baseline: Annotated[Path | None, typer.Option(help="Baseline run to compare with")] = None,
    html: Annotated[Path | None, typer.Option(help="Write the HTML report here")] = Path(
        "ragcheck-report.html"
    ),
    md: Annotated[Path | None, typer.Option(help="Write a Markdown summary here (for PR comments)")] = None,
    save_as_baseline: Annotated[
        bool, typer.Option("--save-baseline", help="Make this run the new baseline")
    ] = False,
    no_judge: Annotated[bool, typer.Option("--no-judge", help="Skip LLM-judged metrics")] = False,
) -> None:
    """Ask every golden question, score the answers, and compare with the baseline."""
    cfg = _load(config)
    if no_judge:
        cfg.judge.provider = "none"
    try:
        cases = load_golden(cfg.path(cfg.golden_set))
        target = make_target(cfg)
        judge = make_judge(cfg)
    except (ConfigError, JudgeError) as e:
        _die(str(e))

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
            return await run_pipeline(cfg, cases, target, judge, label_, tick)
        finally:
            await target.aclose()
            if judge:
                await judge.aclose()

    typer.echo(f"RAGCheck {__version__} · {len(cases)} questions · judge: {judge.name if judge else 'none'}")
    result = asyncio.run(go())
    typer.echo("")

    workdir = cfg.path(cfg.workdir)
    saved = save_run(result, workdir)
    base_path = baseline or _baseline_path(cfg)
    base = load_run(base_path) if base_path.exists() else None
    if base is None:
        typer.secho("  No baseline yet: only absolute minimums are checked.", fg=typer.colors.YELLOW)

    verdict = evaluate(result, base, cfg.thresholds)
    _print_verdict(result, verdict)
    write_reports(result, verdict, base, html, md)
    typer.echo(f"\n  run saved to {saved}")
    if html:
        typer.echo(f"  report: {html.resolve()}")
    if judge:
        typer.echo(f"  judge cache: {judge.hits} hits, {judge.misses} new calls")

    if save_as_baseline:
        if verdict.passed or base is None:
            save_baseline(result, _baseline_path(cfg))
            typer.echo(f"  saved as baseline: {_baseline_path(cfg)}")
        else:
            typer.secho("  not saved as baseline, because the run failed", fg=typer.colors.YELLOW)

    raise typer.Exit(0 if verdict.passed else EXIT_FAILED)


@app.command()
def compare(
    current: Annotated[Path | None, typer.Argument(help="Run file (default: the latest run)")] = None,
    baseline: Annotated[Path | None, typer.Option(help="Baseline run file")] = None,
    config: ConfigOpt = Path("ragcheck.yaml"),
    html: Annotated[Path | None, typer.Option(help="Write the HTML report here")] = None,
    md: Annotated[Path | None, typer.Option(help="Write a Markdown summary here")] = None,
) -> None:
    """Re-check a saved run against a baseline without calling the pipeline again."""
    cfg = _load(config)
    workdir = cfg.path(cfg.workdir)
    cur_path = current or latest_run(workdir)
    if cur_path is None:
        _die("No runs yet. Run `ragcheck run` first.")
    base_path = baseline or _baseline_path(cfg)
    if not base_path.exists():
        _die(f"Baseline not found: {base_path}. Save one with `ragcheck baseline`.")
    cur, base = load_run(cur_path), load_run(base_path)
    verdict = evaluate(cur, base, cfg.thresholds)
    _print_verdict(cur, verdict)
    write_reports(cur, verdict, base, html, md)
    raise typer.Exit(0 if verdict.passed else EXIT_FAILED)


@app.command("baseline")
def set_baseline(
    run_file: Annotated[Path | None, typer.Argument(help="Run file (default: the latest run)")] = None,
    config: ConfigOpt = Path("ragcheck.yaml"),
) -> None:
    """Make a saved run the baseline that future runs are compared with."""
    cfg = _load(config)
    src = run_file or latest_run(cfg.path(cfg.workdir))
    if src is None:
        _die("No runs yet. Run `ragcheck run` first.")
    save_baseline(load_run(src), _baseline_path(cfg))
    typer.echo(f"Baseline set to {src.name} → {_baseline_path(cfg)}")


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
