from __future__ import annotations

import os

import httpx

from ragcheck.config import JudgeConfig
from ragcheck.judges import JudgeError


class AnthropicJudge:
    def __init__(self, cfg: JudgeConfig) -> None:
        env = cfg.api_key_env or "ANTHROPIC_API_KEY"
        key = os.environ.get(env)
        if not key:
            raise JudgeError(f"Set {env} to use the anthropic judge")
        self.model = cfg.model
        self.name = f"anthropic/{cfg.model}"
        self.client = httpx.AsyncClient(
            base_url=cfg.base_url or "https://api.anthropic.com",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            timeout=cfg.timeout_s,
        )

    async def complete(self, system: str, prompt: str) -> str:
        r = await self.client.post(
            "/v1/messages",
            json={
                "model": self.model,
                "max_tokens": 1024,
                "temperature": 0,
                "system": system + " Reply with a single JSON object and nothing else.",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        blocks = r.json()["content"]
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    async def aclose(self) -> None:
        await self.client.aclose()
