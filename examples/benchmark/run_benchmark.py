"""How fast is RAGCheck itself? Runs N questions through the real runner and scoring, with an instant fake
pipeline and an instant fake judge, so the time measured is RAGCheck's own overhead.

    python examples/benchmark/run_benchmark.py            # 50,000 questions
    python examples/benchmark/run_benchmark.py --n 5000

Peak memory needs psutil (`pip install psutil`); without it, only time is reported.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from ragcheck.config import Config, JudgeConfig
from ragcheck.judges import CachedJudge
from ragcheck.models import Context, GoldenCase, TargetResponse
from ragcheck.runner import run


class InstantTarget:
    async def ask(self, question: str) -> TargetResponse:
        n = question.rsplit(" ", 1)[-1]
        return TargetResponse(
            answer=f"Document {n} says the refund window is 30 days.",
            contexts=[Context(id=f"doc-{n}.md#0", text=f"Document {n}: the refund window is 30 days.")],
        )

    async def aclose(self) -> None:
        pass


class InstantJudge:
    """Answers every judge prompt at once, with a quote that really is in the context."""

    name = "instant"

    async def complete(self, system: str, prompt: str, schema: Any = None, sample: int = 0) -> str:
        if '"claims"' in prompt:
            return json.dumps(
                {
                    "claims": [
                        {"claim": "30 days", "supported": True, "evidence": "the refund window is 30 days"}
                    ]
                }
            )
        if '"declined"' in prompt:
            return json.dumps({"declined": False})
        return json.dumps({"score": 5, "reason": "ok"})

    async def aclose(self) -> None:
        pass


def peak_memory_mb(stop: threading.Event, out: list[float]) -> None:
    try:
        import psutil
    except ImportError:
        return
    proc, peak = psutil.Process(), 0
    while not stop.is_set():
        peak = max(peak, proc.memory_info().rss)
        time.sleep(0.05)
    out.append(peak / 2**20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    cases = [
        GoldenCase(
            id=f"q{i}",
            question=f"How long is the refund window in document {i}",
            expected_sources=[f"doc-{i}.md"],
            reference_answer="30 days",
        )
        for i in range(args.n)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config.model_validate(
            {
                "target": {"type": "python", "callable": "x:y"},
                "judge": JudgeConfig(provider="ollama", concurrency=args.concurrency).model_dump(),
                "concurrency": args.concurrency,
            }
        )
        judge = CachedJudge(InstantJudge(), Path(tmp) / "cache", concurrency=args.concurrency)
        stop, peak = threading.Event(), []
        watcher = threading.Thread(target=peak_memory_mb, args=(stop, peak), daemon=True)
        watcher.start()
        start = time.perf_counter()
        result = asyncio.run(run(cfg, cases, InstantTarget(), judge))  # type: ignore[arg-type]
        took = time.perf_counter() - start
        stop.set()
        watcher.join()
        asyncio.run(judge.aclose())

    judged = sum(1 for r in result.results if "faithfulness" in r.scores)
    line = f"{args.n:,} questions ({judged:,} judged on 3 metrics) in {took:.1f} s · {args.n / took:,.0f} questions/s"
    if peak:
        line += f" · peak memory {peak[0]:.0f} MB"
    print(line)


if __name__ == "__main__":
    main()
