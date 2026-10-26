"""Any OpenAI-compatible chat endpoint as the judge: OpenAI, Azure, vLLM, LM Studio, OpenRouter…"""

from __future__ import annotations

import os
from typing import Any

import httpx

from ragcheck.config import JudgeConfig
from ragcheck.judges import FatalJudgeError, JudgeError, Schema, retry_after_s


class OpenAIJudge:
    def __init__(self, cfg: JudgeConfig, model: str) -> None:
        env = cfg.api_key_env or "OPENAI_API_KEY"
        key = os.environ.get(env)
        if not key and not cfg.base_url:
            raise FatalJudgeError(f"Set {env} to use the openai judge")
        self.model = model
        self.name = f"openai/{model}"
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.client = httpx.AsyncClient(
            base_url=cfg.base_url or "https://api.openai.com/v1", headers=headers, timeout=cfg.timeout_s
        )
        # Compatible servers differ in what they accept. Each flag turns off once the server rejects it,
        # so one 400 costs one extra call instead of failing every question.
        self.send_temperature = True
        self.format_mode = "json_schema"  # then "json_object", then "none"

    def _body(self, system: str, prompt: str, schema: Schema | None, sample: int = 0) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        if self.send_temperature:
            body["temperature"] = 0 if sample == 0 else 0.8
        if schema and self.format_mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "judgement", "schema": schema},
            }
        elif self.format_mode in ("json_schema", "json_object"):
            body["response_format"] = {"type": "json_object"}
        return body

    def _downgrade(self, error_text: str) -> bool:
        """Drop the parameter a 400 complains about. False when there is nothing left to drop."""
        text = error_text.lower()
        if self.send_temperature and "temperature" in text:
            self.send_temperature = False
            return True
        if "response_format" in text or "json_schema" in text or "json_object" in text:
            if self.format_mode == "json_schema":
                self.format_mode = "json_object"
                return True
            if self.format_mode == "json_object":
                self.format_mode = "none"
                return True
        return False

    async def complete(self, system: str, prompt: str, schema: Schema | None = None, sample: int = 0) -> str:
        for _ in range(4):  # at most one retry per parameter that can be dropped
            try:
                r = await self.client.post(
                    "/chat/completions", json=self._body(system, prompt, schema, sample)
                )
            except httpx.ConnectError as e:
                raise FatalJudgeError(f"Cannot reach {self.client.base_url}: {e}") from e
            if r.status_code == 400 and self._downgrade(r.text):
                continue
            break
        if r.status_code in (401, 403):
            raise FatalJudgeError(f"The judge endpoint rejected the API key ({r.status_code})")
        if r.status_code == 404:
            raise FatalJudgeError(f"The judge endpoint has no model '{self.model}' (404)")
        if r.status_code == 400:
            raise FatalJudgeError(f"The judge endpoint rejected the request: {r.text[:300]}")
        if r.status_code == 429:
            raise JudgeError("Rate limited (429)", retry_after=retry_after_s(r.headers))
        if r.status_code >= 500:
            raise JudgeError(f"Judge server error {r.status_code}")
        r.raise_for_status()

        try:
            choice = r.json()["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise JudgeError(f"Unexpected reply shape: {r.text[:200]}") from e
        if message.get("refusal"):
            raise JudgeError(f"The judge model refused: {message['refusal'][:200]}", retryable=False)
        if choice.get("finish_reason") == "length":
            raise JudgeError("The judge's reply was cut off (finish_reason=length)", retryable=False)
        content = message.get("content")
        if not content:
            raise JudgeError("The judge returned no content")
        return str(content)

    async def aclose(self) -> None:
        await self.client.aclose()
