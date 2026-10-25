from __future__ import annotations

import httpx

from ragcheck.config import JudgeConfig
from ragcheck.judges import JudgeError


class OllamaJudge:
    """A local model through Ollama: nothing leaves the machine."""

    def __init__(self, cfg: JudgeConfig) -> None:
        self.model = cfg.model
        self.name = f"ollama/{cfg.model}"
        self.client = httpx.AsyncClient(
            base_url=cfg.base_url or "http://localhost:11434", timeout=cfg.timeout_s
        )

    async def complete(self, system: str, prompt: str) -> str:
        try:
            r = await self.client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0, "seed": 7},
                },
            )
        except httpx.ConnectError as e:
            raise JudgeError(
                "Cannot reach Ollama. Start it with `ollama serve`, or set judge.provider."
            ) from e
        if r.status_code == 404:
            raise JudgeError(f"Ollama has no model '{self.model}'. Run `ollama pull {self.model}`.")
        r.raise_for_status()
        return str(r.json()["message"]["content"])

    async def aclose(self) -> None:
        await self.client.aclose()
