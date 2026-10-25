"""Claude as the judge, through the official Anthropic SDK (`pip install ragcheck-ci[anthropic]`)."""

from __future__ import annotations

import os
from typing import Any

from ragcheck.config import JudgeConfig
from ragcheck.judges import FatalJudgeError, JudgeError, Schema, retry_after_s


class AnthropicJudge:
    def __init__(self, cfg: JudgeConfig, model: str) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise FatalJudgeError(
                "The anthropic judge needs the Anthropic SDK: pip install 'ragcheck-ci[anthropic]'"
            ) from e
        self._sdk = anthropic
        self.model = model
        self.name = f"anthropic/{model}"
        self.effort = cfg.effort
        kwargs: dict[str, Any] = {"timeout": cfg.timeout_s, "max_retries": 3}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        if cfg.api_key_env:
            key = os.environ.get(cfg.api_key_env)
            if not key:
                raise FatalJudgeError(f"Set {cfg.api_key_env} to use the anthropic judge")
            kwargs["api_key"] = key
        # Otherwise the SDK finds credentials itself: ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile
        try:
            self.client = anthropic.AsyncAnthropic(**kwargs)
        except anthropic.AnthropicError as e:
            raise FatalJudgeError(f"Anthropic client setup failed: {e}") from e

    async def complete(self, system: str, prompt: str, schema: Schema | None = None) -> str:
        sdk = self._sdk
        # No temperature: the Messages API rejects sampling parameters on current models
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        output_config: dict[str, Any] = {}
        if schema:
            # Structured outputs: the reply is guaranteed to be JSON matching the schema
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if self.effort:
            output_config["effort"] = self.effort
        if output_config:
            params["output_config"] = output_config
        try:
            response = await self.client.messages.create(**params)
        except (sdk.AuthenticationError, sdk.PermissionDeniedError) as e:
            raise FatalJudgeError(f"Anthropic rejected the credentials: {e.message}") from e
        except sdk.NotFoundError as e:
            raise FatalJudgeError(f"Anthropic has no model '{self.model}': {e.message}") from e
        except sdk.BadRequestError as e:
            # The request shape or model settings are wrong: every call would fail the same way
            raise FatalJudgeError(f"Anthropic rejected the request: {e.message}") from e
        except sdk.RateLimitError as e:
            # The SDK already retried; let RAGCheck's own backoff take a turn, honouring Retry-After
            raise JudgeError(
                f"Rate limited: {e.message}", retry_after=retry_after_s(e.response.headers)
            ) from e
        except sdk.APIStatusError as e:
            raise JudgeError(
                f"Anthropic error {e.status_code}: {e.message}", retryable=e.status_code >= 500
            ) from e
        except sdk.APIConnectionError as e:
            raise JudgeError(f"Cannot reach Anthropic: {e}") from e

        if response.stop_reason == "refusal":
            raise JudgeError("The judge model declined to grade this answer", retryable=False)
        if response.stop_reason == "max_tokens":
            raise JudgeError("The judge's reply was cut off at max_tokens", retryable=False)
        text = "".join(b.text for b in response.content if b.type == "text")
        if not text:
            raise JudgeError("The judge returned no text")
        return text

    async def aclose(self) -> None:
        await self.client.close()
