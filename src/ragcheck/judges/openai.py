from __future__ import annotations

import os

import httpx

from ragcheck.config import JudgeConfig
from ragcheck.judges import JudgeError


class OpenAIJudge:
    """Any OpenAI-compatible chat endpoint (OpenAI, Azure, vLLM, LM Studio…)."""

    def __init__(self, cfg: JudgeConfig) -> None:
        env = cfg.api_key_env or "OPENAI_API_KEY"
        key = os.environ.get(env)
        if not key:
            raise JudgeError(f"Set {env} to use the openai judge")
        self.model = cfg.model
        self.name = f"openai/{cfg.model}"
        self.client = httpx.AsyncClient(
            base_url=cfg.base_url or "https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {key}"},
            timeout=cfg.timeout_s,
        )

    async def complete(self, system: str, prompt: str) -> str:
        r = await self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        return str(r.json()["choices"][0]["message"]["content"])

    async def aclose(self) -> None:
        await self.client.aclose()
