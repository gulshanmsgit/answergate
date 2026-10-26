"""Scale: worker pool, rate limiting, checkpoint/resume, sharding, sampling, merging, and a 5,000-question run."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
from conftest import quote_from_prompt
from typer.testing import CliRunner

from ragcheck.baseline import evaluate
from ragcheck.cli import app
from ragcheck.config import Config, ConfigError, Thresholds
from ragcheck.judges import CachedJudge
from ragcheck.models import GoldenCase, TargetResponse
from ragcheck.report import html
from ragcheck.runner import Checkpoint, run
from ragcheck.selection import select, shard_of

cli = CliRunner()


def cases(n: int) -> list[GoldenCase]:
    return [
        GoldenCase(
            id=f"q{i:05d}", question=f"question {i}", expected_sources=[f"doc{i % 50}.md"], tags=[f"t{i % 3}"]
        )
        for i in range(n)
    ]


def cfg(**extra: Any) -> Config:
    return Config.model_validate(
        {"target": {"type": "python", "callable": "a:b"}, "judge": {"provider": "none"}, **extra}
    )


class FastTarget:
    """Answers instantly, tracking how many calls are in flight at once."""

    def __init__(self, delay: float = 0.0, fail_after: int | None = None) -> None:
        self.delay, self.fail_after = delay, fail_after
        self.active = self.peak = self.calls = 0
        self.asked: list[str] = []

    async def ask(self, question: str) -> TargetResponse:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise Interrupted()
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        self.asked.append(question)
        i = int(question.split()[-1])
        return TargetResponse.model_validate(
            {"answer": f"Answer {i}.", "contexts": [{"id": f"doc{i % 50}.md#0", "text": f"Answer {i}."}]}
        )

    async def aclose(self) -> None:
        return None


class Interrupted(BaseException):
    """Stands in for Ctrl+C in the middle of a run."""


# ── Selection ────────────────────────────────────────────────────────


def test_shards_partition_the_set_exactly_and_stably() -> None:
    cs = cases(1000)
    parts = [select(cs, shard=f"{i}/4") for i in range(1, 5)]
    ids = [c.id for p in parts for c in p]
    assert sorted(ids) == sorted(c.id for c in cs)  # every question exactly once
    assert all(200 < len(p) < 300 for p in parts)  # roughly even
    assert [c.id for c in select(list(reversed(cs)), shard="3/4")][::-1] == [c.id for c in parts[2]]
    assert shard_of("q00042", 4) == shard_of("q00042", 4)


def test_sample_is_seeded_and_keeps_order() -> None:
    cs = cases(500)
    a, b = select(cs, sample=50, seed=7), select(cs, sample=50, seed=7)
    assert [c.id for c in a] == [c.id for c in b] and len(a) == 50
    assert [c.id for c in a] == sorted(c.id for c in a)
    assert [c.id for c in select(cs, sample=50, seed=8)] != [c.id for c in a]


def test_tags_and_bad_specs() -> None:
    cs = cases(30)
    assert all("t1" in c.tags for c in select(cs, tags=["t1"]))
    for bad in ("0/4", "5/4", "two/four"):
        with pytest.raises(ConfigError):
            select(cs, shard=bad)
    with pytest.raises(ConfigError, match="No golden questions"):
        select(cs, tags=["missing"])


# ── Runner ───────────────────────────────────────────────────────────


def test_worker_pool_respects_concurrency() -> None:
    t = FastTarget(delay=0.002)
    r = asyncio.run(run(cfg(concurrency=8), cases(200), t, None))
    assert len(r.results) == 200 and r.errors == 0
    assert t.peak <= 8
    assert [x.case.id for x in r.results] == [c.id for c in cases(200)]  # golden-set order kept


def test_rate_limit_spaces_calls() -> None:
    start = time.perf_counter()
    asyncio.run(run(cfg(concurrency=10, rate_limit_rps=50), cases(21), FastTarget(), None))
    # 20 gaps of 20 ms, less one tick of the Windows timer that asyncio.sleep wakes on
    assert time.perf_counter() - start >= 0.38 - 0.016


def test_interrupted_run_resumes_where_it_stopped(tmp_path: Path) -> None:
    ckpt = tmp_path / "inprogress.jsonl"
    c = cfg(concurrency=4)
    with pytest.raises(Interrupted):
        asyncio.run(run(c, cases(100), FastTarget(fail_after=60), None, checkpoint=ckpt))
    saved = len(Checkpoint.load(ckpt, c.fingerprint()))
    assert 50 <= saved <= 60

    second = FastTarget()
    r = asyncio.run(run(c, cases(100), second, None, checkpoint=ckpt, resume=True))
    assert len(r.results) == 100 and r.errors == 0
    assert second.calls == 100 - saved  # only the unfinished questions were asked again


def test_resume_ignores_a_checkpoint_from_other_settings(tmp_path: Path) -> None:
    ckpt = tmp_path / "inprogress.jsonl"
    asyncio.run(run(cfg(k=5), cases(10), FastTarget(), None, checkpoint=ckpt))
    assert Checkpoint.load(ckpt, cfg(k=3).fingerprint()) == {}


# ── CLI: shards, merge, subsets ──────────────────────────────────────


def test_shards_then_merge_gives_one_verdict(project: Path) -> None:
    assert cli.invoke(app, ["run", "--save-baseline"]).exit_code == 0
    runs = project / ".ragcheck" / "runs"
    for i in (1, 2):
        r = cli.invoke(app, ["run", "--shard", f"{i}/2"])
        assert r.exit_code == 0, r.output
    shard_files = sorted(runs.glob("*.json"))[-2:]
    r = cli.invoke(app, ["merge", *map(str, shard_files), "--md", "merged.md"])
    assert r.exit_code == 0, r.output
    assert "Merged 2 runs · 4 questions" in r.output
    # Overlapping shards are refused
    r = cli.invoke(app, ["merge", str(shard_files[0]), str(shard_files[0])])
    assert r.exit_code == 2 and "must not overlap" in r.output


def test_partial_runs_cannot_become_the_baseline(project: Path) -> None:
    r = cli.invoke(app, ["run", "--sample", "2", "--save-baseline"])
    assert r.exit_code == 2 and "whole golden set" in r.output


# ── 5,000 questions end to end ───────────────────────────────────────


class InstantJudge:
    name = "instant"

    async def complete(self, system: str, prompt: str, schema: Any = None, sample: int = 0) -> str:
        if "Split the answer" in prompt:
            return json.dumps(
                {"claims": [{"claim": "a", "supported": True, "evidence": quote_from_prompt(prompt)}]}
            )
        return json.dumps({"score": 4, "reason": "", "declined": False})

    async def aclose(self) -> None:
        return None


def test_five_thousand_questions(tmp_path: Path) -> None:
    n = 5000
    c = cfg(
        concurrency=64,
        judge={"provider": "ollama", "concurrency": 64},
        metrics=["hit_rate", "ndcg", "faithfulness", "relevance"],
    )
    judge = CachedJudge(InstantJudge(), tmp_path / "cache", concurrency=64)
    start = time.perf_counter()
    r = asyncio.run(run(c, cases(n), FastTarget(), judge, checkpoint=tmp_path / "ck.jsonl"))
    elapsed = time.perf_counter() - start
    judge.cache.close()

    assert len(r.results) == n and r.errors == 0 and r.judge_errors == 0
    assert r.metrics["hit_rate"] == 1.0 and r.metrics["faithfulness"] == 1.0
    assert elapsed < 60, f"{n} questions took {elapsed:.1f}s"

    # The gate (normal-approximation intervals) and a capped report stay fast at this size
    start = time.perf_counter()
    v = evaluate(r, r, Thresholds())
    page = html(r, v, r, limit=500)
    assert time.perf_counter() - start < 10
    assert v.passed and page.count('class="case"') == 500 and "4500 more passing" in page
