"""A local model through Ollama as the judge: nothing leaves the machine."""

from __future__ import annotations

from typing import Any

import httpx

from ragcheck.config import JudgeConfig
from ragcheck.judges import FatalJudgeError, JudgeError, Schema


class OllamaJudge:
    def __init__(self, cfg: JudgeConfig, model: str) -> None:
        self.model = model
        self.name = f"ollama/{model}"
        self.client = httpx.AsyncClient(
            base_url=cfg.base_url or "http://localhost:11434", timeout=cfg.timeout_s
        )
        # Recent Ollama versions take a JSON schema in "format"; older ones only "json"
        self.schema_format = True

    async def complete(self, system: str, prompt: str, schema: Schema | None = None, sample: int = 0) -> str:
        for _ in range(2):
            fmt: Any = schema if schema and self.schema_format else "json"
            try:
                r = await self.client.post(
                    "/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        "format": fmt,
                        "stream": False,
                        # Sample 0 is deterministic; extra samples vary so votes are independent
                        "options": {"temperature": 0 if sample == 0 else 0.8, "seed": 7 + sample},
                    },
                )
            except httpx.ConnectError as e:
                raise FatalJudgeError(
                    "Cannot reach Ollama. Start it with `ollama serve`, or pick another judge.provider."
                ) from e
            if r.status_code in (400, 500) and isinstance(fmt, dict) and "format" in r.text.lower():
                self.schema_format = False
                continue
            break
        if r.status_code == 404:
            raise FatalJudgeError(f"Ollama has no model '{self.model}'. Run `ollama pull {self.model}`.")
        if r.status_code >= 500:
            raise JudgeError(f"Ollama error {r.status_code}: {r.text[:200]}")
        r.raise_for_status()
        try:
            return str(r.json()["message"]["content"])
        except (ValueError, KeyError, TypeError) as e:
            raise JudgeError(f"Unexpected Ollama reply: {r.text[:200]}") from e

    async def aclose(self) -> None:
        await self.client.aclose()
