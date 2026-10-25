"""Ask every golden question, score every answer, and aggregate the run."""

from __future__ import annotations

import asyncio
import statistics
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone

from ragcheck import guardrails as g
from ragcheck.config import Config
from ragcheck.judges import CachedJudge, JudgeError
from ragcheck.metrics import judged
from ragcheck.metrics.retrieval import retrieval_scores
from ragcheck.models import CaseResult, GoldenCase, RunResult
from ragcheck.targets import Target

Progress = Callable[[CaseResult], None]


async def _ask(target: Target, case: GoldenCase, retries: int) -> tuple[CaseResult, float]:
    delay = 0.5
    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            resp = await target.ask(case.question)
            ms = (time.perf_counter() - start) * 1000
            return CaseResult(case=case, answer=resp.answer, contexts=resp.contexts, latency_ms=ms), ms
        except Exception as e:
            if attempt == retries:
                return CaseResult(case=case, error=f"{type(e).__name__}: {e}"), 0.0
            await asyncio.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


async def score_case(res: CaseResult, cfg: Config, judge: CachedJudge | None) -> None:
    case, wanted = res.case, set(cfg.metrics)

    for name, value in retrieval_scores(res.contexts, case.expected_sources, cfg.k).items():
        if name in wanted:
            res.scores[name] = value

    # Guardrails that need no judge
    pii = g.find_pii(res.answer, cfg.pii_allow)
    res.guardrails["pii"] = not pii
    if pii:
        res.notes["pii"] = "Answer contains: " + ", ".join(pii)
    if case.canary:
        leaked = g.leaked_canary(res.answer, case.canary)
        res.guardrails["injection"] = not leaked
        if leaked:
            res.notes["injection"] = f"Answer followed a planted instruction (canary {case.canary})"

    if case.expect_refusal:
        refused = (
            await judged.is_refusal(judge, case.question, res.answer)
            if judge
            else g.looks_like_refusal(res.answer)
        )
        res.guardrails["refusal"] = refused
        if not refused:
            res.notes["refusal"] = "Answered a question it should have declined"
        # Answer-quality metrics don't apply to a question the pipeline should refuse
        return

    if judge is None:
        return
    try:
        if "faithfulness" in wanted:
            res.scores["faithfulness"], res.notes["faithfulness"] = await judged.faithfulness(
                judge, res.answer, res.contexts[: cfg.k]
            )
        if "relevance" in wanted:
            res.scores["relevance"], res.notes["relevance"] = await judged.relevance(
                judge, case.question, res.answer
            )
        if "correctness" in wanted and case.reference_answer:
            res.scores["correctness"], res.notes["correctness"] = await judged.correctness(
                judge, case.question, res.answer, case.reference_answer
            )
    except JudgeError as e:
        res.error = f"Judge: {e}"


def aggregate(results: list[CaseResult]) -> tuple[dict[str, float], dict[str, float]]:
    metrics: dict[str, list[float]] = {}
    rails: dict[str, list[bool]] = {}
    for r in results:
        for k, v in r.scores.items():
            metrics.setdefault(k, []).append(v)
        for k, ok in r.guardrails.items():
            rails.setdefault(k, []).append(ok)
    return (
        {k: round(statistics.fmean(v), 4) for k, v in sorted(metrics.items())},
        {k: round(sum(v) / len(v), 4) for k, v in sorted(rails.items())},
    )


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, round(p * (len(s) - 1)))]


async def run(
    cfg: Config,
    cases: list[GoldenCase],
    target: Target,
    judge: CachedJudge | None,
    label: str = "",
    on_case: Progress | None = None,
) -> RunResult:
    sem = asyncio.Semaphore(max(1, cfg.concurrency))

    async def one(case: GoldenCase) -> CaseResult:
        async with sem:
            res, _ = await _ask(target, case, cfg.retries)
            if res.error is None:
                await score_case(res, cfg, judge)
        if on_case:
            on_case(res)
        return res

    results = await asyncio.gather(*(one(c) for c in cases))
    metrics, rails = aggregate(results)
    latencies = [r.latency_ms for r in results if r.error is None]
    now = datetime.now(timezone.utc)
    return RunResult(
        id=now.strftime("%Y%m%d-%H%M%S-%f-") + uuid.uuid4().hex[:4],
        created_at=now.isoformat(timespec="seconds"),
        label=label,
        config_hash=cfg.fingerprint(),
        results=list(results),
        metrics=metrics,
        guardrails=rails,
        latency_p50_ms=round(_percentile(latencies, 0.5), 1),
        latency_p95_ms=round(_percentile(latencies, 0.95), 1),
        errors=sum(1 for r in results if r.error),
    )
