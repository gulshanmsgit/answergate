"""LLM judges. Every judge call goes through a disk cache so reruns are cheap and repeatable."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Protocol

from ragcheck.config import Config


class JudgeError(Exception):
    pass


class Judge(Protocol):
    name: str

    async def complete(self, system: str, prompt: str) -> str: ...

    async def aclose(self) -> None: ...


def parse_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (tolerates code fences and chatter)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise JudgeError(f"Judge did not return JSON: {text[:200]!r}")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise JudgeError(f"Judge returned invalid JSON: {text[:200]!r}") from e
    if not isinstance(data, dict):
        raise JudgeError("Judge JSON must be an object")
    return data


class CachedJudge:
    """Wraps a judge with a content-addressed cache in <workdir>/cache."""

    def __init__(self, inner: Judge, cache_dir: Path) -> None:
        self.inner = inner
        self.name = inner.name
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    async def ask_json(self, system: str, prompt: str) -> dict[str, Any]:
        key = hashlib.sha256(f"{self.name}\0{system}\0{prompt}".encode()).hexdigest()
        path = self.dir / f"{key[:2]}" / f"{key}.json"
        if path.exists():
            self.hits += 1
            return dict(json.loads(path.read_text(encoding="utf-8")))
        self.misses += 1
        data = parse_json(await self.inner.complete(system, prompt))
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return data

    async def aclose(self) -> None:
        await self.inner.aclose()


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
    return CachedJudge(inner, cfg.path(cfg.workdir) / "cache")
