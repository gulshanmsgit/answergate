"""LLM judges. Every judge call goes through a disk cache so reruns are cheap and repeatable."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Protocol

import httpx

from ragcheck.config import Config, JudgeConfig


class JudgeError(Exception):
    pass


class Judge(Protocol):
    name: str

    async def complete(self, system: str, prompt: str) -> str: ...

    async def aclose(self) -> None: ...


def parse_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (tolerates code fences and chatter around it)."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise JudgeError(f"Judge did not return a JSON object: {text[:200]!r}")


def _transient(e: Exception) -> bool:
    """Worth retrying: bad JSON from a sampling hiccup, timeouts, rate limits and server errors."""
    if isinstance(e, JudgeError | httpx.TimeoutException | httpx.NetworkError):
        return True
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code == 429 or e.response.status_code >= 500
    return False


class CachedJudge:
    """Wraps a judge with retries and a content-addressed cache in <workdir>/cache."""

    def __init__(
        self,
        inner: Judge,
        cache_dir: Path,
        attempts: int = 3,
        backoff_s: float = 0.5,
        concurrency: int = 2,
        identity: str | None = None,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        # What the cache key is tied to: provider, model and endpoint, not just the model name
        self.identity = identity or inner.name
        self.sem = asyncio.Semaphore(max(1, concurrency))
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.attempts = max(1, attempts)
        self.backoff_s = backoff_s
        self.hits = 0
        self.misses = 0

    def _read(self, path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A half-written or corrupted entry is just a cache miss
            return None
        return data if isinstance(data, dict) else None

    def _write(self, path: Path, data: dict[str, Any]) -> None:
        # Write then rename, so a crash never leaves a truncated entry behind
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            path.parent.mkdir(exist_ok=True)
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            # Caching is an optimisation: a locked or read-only cache must not fail the run
            tmp.unlink(missing_ok=True)

    async def ask_json(self, system: str, prompt: str) -> dict[str, Any]:
        key = hashlib.sha256(f"{self.identity}\0{system}\0{prompt}".encode()).hexdigest()
        path = self.dir / key[:2] / f"{key}.json"
        cached = self._read(path) if path.exists() else None
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        delay = self.backoff_s
        for attempt in range(1, self.attempts + 1):
            try:
                async with self.sem:
                    reply = await self.inner.complete(system, prompt)
                data = parse_json(reply)
                break
            except Exception as e:
                if attempt == self.attempts or not _transient(e):
                    raise JudgeError(f"{type(e).__name__}: {e}"[:300]) from e
                await asyncio.sleep(delay)
                delay *= 2
        self._write(path, data)
        return data

    async def aclose(self) -> None:
        await self.inner.aclose()


def cache_identity(j: JudgeConfig) -> str:
    return f"{j.provider}|{j.model}|{j.base_url or 'default'}"


def make_judge(cfg: Config) -> CachedJudge | None:
    j = cfg.judge
    if j.provider == "none":
        return None
    inner: Judge
    if j.provider == "ollama":
        from ragcheck.judges.ollama import OllamaJudge

        inner = OllamaJudge(j)
    elif j.provider == "openai":
        from ragcheck.judges.openai import OpenAIJudge

        inner = OpenAIJudge(j)
    else:
        from ragcheck.judges.anthropic import AnthropicJudge

        inner = AnthropicJudge(j)
    return CachedJudge(
        inner, cfg.path(cfg.workdir) / "cache", concurrency=j.concurrency, identity=cache_identity(j)
    )
