"""Choosing which golden questions a run asks: tags, deterministic shards and seeded samples."""

from __future__ import annotations

import hashlib
import random
import re

from ragcheck.config import ConfigError
from ragcheck.models import GoldenCase


def parse_shard(spec: str) -> tuple[int, int]:
    """'2/4' -> (2, 4): the second of four shards."""
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", spec)
    if not m:
        raise ConfigError(f"--shard must look like 2/4, got '{spec}'")
    i, n = int(m.group(1)), int(m.group(2))
    if not 1 <= i <= n:
        raise ConfigError(f"--shard {spec}: the shard number must be between 1 and {n}")
    return i, n


def shard_of(case_id: str, n: int) -> int:
    """Stable shard (1..n) from a hash of the id: the same question always lands on the same shard,
    whatever order the golden set is in and whichever machine runs it."""
    digest = hashlib.sha256(case_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % n + 1


def select(
    cases: list[GoldenCase],
    shard: str | None = None,
    sample: int | None = None,
    seed: int = 0,
    tags: list[str] | None = None,
) -> list[GoldenCase]:
    chosen = cases
    if tags:
        wanted = set(tags)
        chosen = [c for c in chosen if wanted & set(c.tags)]
        if not chosen:
            raise ConfigError(f"No golden questions have the tag(s): {', '.join(sorted(wanted))}")
    if shard:
        i, n = parse_shard(shard)
        chosen = [c for c in chosen if shard_of(c.id, n) == i]
    if sample is not None:
        if sample < 1:
            raise ConfigError("--sample must be at least 1")
        if sample < len(chosen):
            # Seeded, and keeps golden-set order, so the same seed gives the same questions
            picked = set(random.Random(seed).sample(range(len(chosen)), sample))
            chosen = [c for i, c in enumerate(chosen) if i in picked]
    return chosen
