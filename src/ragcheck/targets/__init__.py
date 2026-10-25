"""Targets: how RAGCheck calls the pipeline under test."""

from __future__ import annotations

from typing import Protocol

from ragcheck.config import Config, HttpTargetConfig
from ragcheck.models import TargetResponse


class Target(Protocol):
    async def ask(self, question: str) -> TargetResponse: ...

    async def aclose(self) -> None: ...


def make_target(cfg: Config) -> Target:
    if isinstance(cfg.target, HttpTargetConfig):
        from ragcheck.targets.http import HttpTarget

        return HttpTarget(cfg.target)
    from ragcheck.targets.python import PythonTarget

    return PythonTarget(cfg.target, cfg.root)
