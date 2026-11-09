"""Ask every golden question, score every answer, and aggregate the run."""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

from answergate import guardrails as g
from answergate.config import Config
from answergate.judges import CachedJudge, FatalJudgeError, JudgeError
from answergate.metrics import judged
from answergate.metrics.numbers import score_numbers
from answergate.metrics.retrieval import retrieval_scores
from answergate.models import CaseResult, GoldenCase, RunResult
from answergate.targets import Target

Progress = Callable[[CaseResult], None]


async def _ask(target: Target, case: GoldenCase, retries: int, timeout_s: float) -> CaseResult:
    delay = 0.5
    error = ""
    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            resp = await asyncio.wait_for(target.ask(case.question), timeout_s)
            ms = (time.perf_counter() - start) * 1000
            return CaseResult(
                case=case,
                answer=resp.answer,
                contexts=resp.contexts,
                latency_ms=ms,
                pipeline=resp.pipeline,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
            )
        except asyncio.TimeoutError:
            error = f"TimeoutError: no answer within {timeout_s:g}s"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"[:500]
        if attempt < retries:
            await asyncio.sleep(delay)
            delay *= 2
    tries = f" (after {retries + 1} attempts)" if retries else ""
    return CaseResult(case=case, error=error + tries)


_NEGATION = re.compile(r"\b(not|no|never|none|cannot|can't|don't|doesn't|isn't|aren't|won't|without)\b")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def answer_overlap(a: str, b: str) -> float:
    """Word-overlap F1 between two answers: 1.0 for the same words, near 0 for unrelated ones.
    Answers that disagree on a number or on a negation share most words but not the fact, so they
    count as at most half consistent ("600 seconds" vs "60 seconds", "is allowed" vs "is not allowed")."""
    a, b = a.lower(), b.lower()
    ta, tb = Counter(re.findall(r"\w+", a)), Counter(re.findall(r"\w+", b))
    if not ta and not tb:
        return 1.0
    common = sum((ta & tb).values())
    if not common:
        return 0.0
    p, r = common / sum(ta.values()), common / sum(tb.values())
    score = 2 * p * r / (p + r)
    if set(_NUMBER.findall(a)) != set(_NUMBER.findall(b)) or bool(_NEGATION.search(a)) != bool(
        _NEGATION.search(b)
    ):
        score = min(score, 0.5)
    return score


def consistency(answers: list[str]) -> float:
    """Mean overlap over every pair of answers to the same question."""
    pairs = list(itertools.combinations(answers, 2))
    return round(statistics.fmean(answer_overlap(a, b) for a, b in pairs), 4) if pairs else 1.0


async def score_case(res: CaseResult, cfg: Config, judge: CachedJudge | None) -> None:
    case, wanted = res.case, set(cfg.metrics)

    for name, value in retrieval_scores(res.contexts, case.expected_sources, cfg.k, cfg.source_match).items():
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
        refused = g.looks_like_refusal(res.answer)
        if judge:
            try:
                refused = await judged.is_refusal(judge, case.question, res.answer, cfg.judge.samples)
            except JudgeError as e:
                # The heuristic above stands in, and the run records that the judge failed
                res.judge_error = f"refusal: {e}"
        res.guardrails["refusal"] = refused
        if not refused:
            res.notes["refusal"] = "Answered a question it should have declined"
        # Answer-quality metrics don't apply to a question the pipeline should refuse
        return

    # Figures checked without a judge: a wrong rate or limit fails here even if the wording is right
    if "numbers" in wanted and case.reference_answer:
        checked = score_numbers(res.answer, case.reference_answer)
        if checked is not None:
            res.scores["numbers"], res.notes["numbers"] = checked

    if judge is None:
        return
    n, top = cfg.judge.samples, res.contexts[: cfg.k]
    checks: list[tuple[str, Callable[[], Awaitable[judged.Judged]]]] = []
    if "faithfulness" in wanted:
        verify = cfg.judge.verify_evidence
        checks.append(("faithfulness", lambda: judged.faithfulness(judge, res.answer, top, n, verify)))
    if "relevance" in wanted:
        checks.append(("relevance", lambda: judged.relevance(judge, case.question, res.answer, n)))
    if "correctness" in wanted and case.reference_answer:
        ref = case.reference_answer
        checks.append(("correctness", lambda: judged.correctness(judge, case.question, res.answer, ref, n)))
    failed: list[str] = []
    # Each metric stands alone: one bad judge reply loses that score, not the others
    for name, check in checks:
        try:
            result = await check()
        except JudgeError as e:
            failed.append(f"{name}: {e}")
            continue
        res.scores[name], res.notes[name] = result.score, result.note
        if n > 1:
            res.judge_spread[name] = result.spread
    if failed:
        res.judge_error = "; ".join(failed)[:500]


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


def run_pipeline_settings(results: list[CaseResult]) -> dict[str, str]:
    """Each setting's most common reported value; a setting that varied between answers says so."""
    seen: dict[str, Counter[str]] = {}
    for r in results:
        for k, v in r.pipeline.items():
            seen.setdefault(k, Counter())[v] += 1
    out: dict[str, str] = {}
    for k, counts in sorted(seen.items()):
        value = counts.most_common(1)[0][0]
        out[k] = value if len(counts) == 1 else f"{value} (varied: {len(counts)} values)"
    return out


def build_run(
    results: list[CaseResult], config_hash: str, label: str, source_match: str, cost_currency: str = ""
) -> RunResult:
    """Aggregate per-question results into a run. Used after a run, a resume and a shard merge."""
    metrics, rails = aggregate(results)
    latencies = [r.latency_ms for r in results if r.error is None]
    tokens = [
        r.tokens_in + r.tokens_out for r in results if r.tokens_in is not None and r.tokens_out is not None
    ]
    costs = [r.cost for r in results if r.cost is not None]
    now = datetime.now(timezone.utc)
    return RunResult(
        id=now.strftime("%Y%m%d-%H%M%S-%f-") + uuid.uuid4().hex[:4],
        created_at=now.isoformat(timespec="seconds"),
        label=label,
        config_hash=config_hash,
        results=results,
        metrics=metrics,
        guardrails=rails,
        latency_p50_ms=round(_percentile(latencies, 0.5), 1),
        latency_p95_ms=round(_percentile(latencies, 0.95), 1),
        errors=sum(1 for r in results if r.error),
        judge_errors=sum(1 for r in results if r.judge_error),
        source_match=source_match,  # type: ignore[arg-type]
        pipeline=run_pipeline_settings(results),
        tokens_per_question=round(statistics.fmean(tokens), 1) if tokens else None,
        cost_per_question=round(statistics.fmean(costs), 6) if costs else None,
        cost_currency=cost_currency if costs else "",
    )


class RateLimiter:
    """Spaces out call starts to at most `rps` per second (shared by all workers)."""

    def __init__(self, rps: float | None) -> None:
        self.interval = 1.0 / rps if rps else 0.0
        self.next_at = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        if not self.interval:
            return
        async with self.lock:
            now = time.perf_counter()  # monotonic ticks only every ~16 ms on Windows
            delay = self.next_at - now
            self.next_at = max(now, self.next_at) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


class Checkpoint:
    """Appends each finished question to a JSONL file, so a crash or Ctrl+C loses nothing."""

    def __init__(self, path: Path, config_hash: str) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.exists()
        self.f = path.open("a", encoding="utf-8")
        if fresh:
            self.f.write(json.dumps({"config_hash": config_hash}) + "\n")
            self.f.flush()

    def add(self, res: CaseResult) -> None:
        self.f.write(res.model_dump_json() + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()

    @staticmethod
    def load(path: Path, config_hash: str) -> dict[str, CaseResult]:
        """Finished questions from an earlier, interrupted run with the same scoring settings."""
        done: dict[str, CaseResult] = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return done
        if not lines or json.loads(lines[0]).get("config_hash") != config_hash:
            return done
        for line in lines[1:]:
            try:
                r = CaseResult.model_validate_json(line)
            except ValueError:
                continue  # a line cut off by the crash
            # Questions that errored are asked again on resume
            if r.error is None and r.judge_error is None:
                done[r.case.id] = r
        return done


async def run(
    cfg: Config,
    cases: list[GoldenCase],
    target: Target,
    judge: CachedJudge | None,
    label: str = "",
    on_case: Progress | None = None,
    checkpoint: Path | None = None,
    resume: bool = False,
) -> RunResult:
    fingerprint = cfg.fingerprint()
    prior = Checkpoint.load(checkpoint, fingerprint) if (checkpoint and resume) else {}
    if checkpoint and not resume and checkpoint.exists():
        checkpoint.unlink()
    log = Checkpoint(checkpoint, fingerprint) if checkpoint else None
    limiter = RateLimiter(cfg.rate_limit_rps)

    results: dict[str, CaseResult] = {}
    queue: asyncio.Queue[GoldenCase] = asyncio.Queue()
    for c in cases:
        if c.id in prior:
            results[c.id] = prior[c.id]
            if on_case:
                on_case(prior[c.id])
        else:
            queue.put_nowait(c)

    async def one(case: GoldenCase) -> CaseResult:
        await limiter.wait()
        res = await _ask(target, case, cfg.retries, cfg.target_timeout_s)
        if cfg.cost and res.tokens_in is not None and res.tokens_out is not None:
            res.cost = (
                res.tokens_in * cfg.cost.input_per_1k + res.tokens_out * cfg.cost.output_per_1k
            ) / 1000
        if res.error is None and cfg.repeats > 1 and not case.expect_refusal:
            await ask_again(res)
        if res.error is None:
            try:
                await score_case(res, cfg, judge)
            except FatalJudgeError:
                raise
            except Exception as e:
                # A scoring bug must not take down the whole run
                res.judge_error = f"Scoring failed: {type(e).__name__}: {e}"[:500]
        return res

    async def ask_again(res: CaseResult) -> None:
        """Ask the same question repeats - 1 more times and score how much the answers agree."""
        for _ in range(cfg.repeats - 1):
            await limiter.wait()
            again = await _ask(target, res.case, cfg.retries, cfg.target_timeout_s)
            if again.error is not None:
                res.notes["consistency"] = f"Not scored: a repeat failed ({again.error})"[:300]
                return
            res.repeat_answers.append(again.answer)
        score = consistency([res.answer, *res.repeat_answers])
        if "consistency" in cfg.metrics:
            res.scores["consistency"] = score
        n = len(res.repeat_answers) + 1
        res.notes["consistency"] = (
            f"{n} answers to the same question were identical"
            if score == 1.0
            else f"{n} answers to the same question share {score:.0%} of their words"
        )

    async def worker() -> None:
        # A fixed pool of workers pulls from the queue: memory stays flat however many questions there are
        while True:
            try:
                case = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            res = await one(case)
            results[case.id] = res
            if log:
                log.add(res)
            if on_case:
                on_case(res)

    workers = [asyncio.ensure_future(worker()) for _ in range(min(cfg.concurrency, max(1, queue.qsize())))]
    try:
        await asyncio.gather(*workers)
    except BaseException:
        # A fatal judge error (or Ctrl+C): stop the other workers; the checkpoint keeps what finished
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise
    finally:
        if log:
            log.close()

    ordered = [results[c.id] for c in cases if c.id in results]
    return build_run(ordered, fingerprint, label, cfg.source_match, cfg.cost.currency if cfg.cost else "")
