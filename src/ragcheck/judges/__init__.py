"""LLM judges. Every judge call goes through retries, a concurrency limit and a disk cache."""

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

# A JSON schema for the judge's reply; each provider enforces it in its own way
Schema = dict[str, Any]

# Upper bound on how long a Retry-After header may pause us
MAX_RETRY_AFTER_S = 60.0


class JudgeError(Exception):
    """One judge call failed. The question loses that score; the run goes on."""

    def __init__(self, message: str, retryable: bool = True, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class FatalJudgeError(Exception):
    """The judge can't work at all (bad key, unknown model, unreachable server, rejected request).
    Every call would fail the same way, so the run stops with exit code 2 instead of grading nothing."""


class Judge(Protocol):
    name: str

    async def complete(
        self, system: str, prompt: str, schema: Schema | None = None, sample: int = 0
    ) -> str: ...

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


def retry_after_s(headers: httpx.Headers | Any) -> float | None:
    """Seconds from a Retry-After header (numeric form), capped so a huge value can't stall the run."""
    try:
        value = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return min(max(value, 0.0), MAX_RETRY_AFTER_S)


def _transient(e: Exception) -> bool:
    """Worth retrying: bad JSON from a sampling hiccup, timeouts, rate limits and server errors."""
    if isinstance(e, JudgeError):
        return e.retryable
    if isinstance(e, httpx.TimeoutException | httpx.NetworkError):
        return True
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code == 429 or e.response.status_code >= 500
    return False


def _delay_for(e: Exception, fallback: float) -> float:
    if isinstance(e, JudgeError) and e.retry_after is not None:
        return e.retry_after
    if isinstance(e, httpx.HTTPStatusError):
        return retry_after_s(e.response.headers) or fallback
    return fallback


PREFLIGHT_SCHEMA: Schema = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


class CachedJudge:
    """Wraps a judge with retries, a concurrency limit and a content-addressed cache in <workdir>/cache."""

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

    async def _call(self, system: str, prompt: str, schema: Schema | None, sample: int = 0) -> dict[str, Any]:
        delay = self.backoff_s
        for attempt in range(1, self.attempts + 1):
            try:
                async with self.sem:
                    reply = await self.inner.complete(system, prompt, schema, sample)
                return parse_json(reply)
            except FatalJudgeError:
                raise
            except Exception as e:
                if attempt == self.attempts or not _transient(e):
                    raise JudgeError(f"{type(e).__name__}: {e}"[:300], retryable=False) from e
                await asyncio.sleep(_delay_for(e, delay))
                delay *= 2
        raise AssertionError("unreachable")

    async def ask_json(
        self, system: str, prompt: str, schema: Schema | None = None, sample: int = 0
    ) -> dict[str, Any]:
        schema_key = json.dumps(schema, sort_keys=True) if schema else ""
        if sample:
            schema_key += f"#sample{sample}"
        key = hashlib.sha256(f"{self.identity}\0{system}\0{prompt}\0{schema_key}".encode()).hexdigest()
        path = self.dir / key[:2] / f"{key}.json"
        cached = self._read(path) if path.exists() else None
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        data = await self._call(system, prompt, schema, sample)
        self._write(path, data)
        return data

    async def preflight(self) -> None:
        """One tiny uncached call before the run, so a bad key or model fails in seconds, not per question."""
        try:
            await self._call(
                "You are a health check. Reply in JSON.", 'Reply with {"ok": true}.', PREFLIGHT_SCHEMA
            )
        except FatalJudgeError:
            raise
        except JudgeError as e:
            raise FatalJudgeError(f"The judge ({self.name}) is not answering usably: {e}") from e

    async def aclose(self) -> None:
        await self.inner.aclose()


# Per-provider default model when judge.model is left out
DEFAULT_MODELS = {"ollama": "qwen2.5:7b", "anthropic": "claude-opus-5-5"}


def resolve_model(j: JudgeConfig) -> str:
    if j.model:
        return j.model
    if j.provider in DEFAULT_MODELS:
        return DEFAULT_MODELS[j.provider]
    raise FatalJudgeError(f"Set judge.model for the {j.provider} judge, e.g. the model your endpoint serves.")


def cache_identity(j: JudgeConfig) -> str:
    return f"{j.provider}|{resolve_model(j)}|{j.base_url or 'default'}|{j.effort or ''}"


def make_judge(cfg: Config) -> CachedJudge | None:
    j = cfg.judge
    if j.provider == "none":
        return None
    model = resolve_model(j)
    inner: Judge
    if j.provider == "ollama":
        from ragcheck.judges.ollama import OllamaJudge

        inner = OllamaJudge(j, model)
    elif j.provider == "openai":
        from ragcheck.judges.openai import OpenAIJudge

        inner = OpenAIJudge(j, model)
    else:
        from ragcheck.judges.anthropic import AnthropicJudge

        inner = AnthropicJudge(j, model)
    return CachedJudge(
        inner, cfg.path(cfg.workdir) / "cache", concurrency=j.concurrency, identity=cache_identity(j)
    )
