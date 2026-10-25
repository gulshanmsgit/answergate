from __future__ import annotations

from typing import Any

import httpx

from ragcheck.config import HttpTargetConfig, expand_env
from ragcheck.models import TargetResponse
from ragcheck.targets.parse import to_response


def _fill(value: Any, question: str) -> Any:
    """Put the question into the request body template."""
    if isinstance(value, str):
        return value.replace("{question}", question)
    if isinstance(value, dict):
        return {k: _fill(v, question) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, question) for v in value]
    return value


class HttpTarget:
    def __init__(self, cfg: HttpTargetConfig) -> None:
        self.cfg = cfg
        headers = {k: expand_env(v) for k, v in cfg.headers.items()}
        self.client = httpx.AsyncClient(headers=headers, timeout=cfg.timeout_s)

    async def ask(self, question: str) -> TargetResponse:
        body = _fill(self.cfg.body, question)
        url = expand_env(self.cfg.url)
        if self.cfg.method == "GET":
            r = await self.client.get(url, params={k: str(v) for k, v in body.items()})
        else:
            r = await self.client.post(url, json=body)
        r.raise_for_status()
        return to_response(
            r.json(),
            self.cfg.answer_path,
            self.cfg.contexts_path,
            self.cfg.context_id_field,
            self.cfg.context_text_field,
        )

    async def aclose(self) -> None:
        await self.client.aclose()
