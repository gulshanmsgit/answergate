"""Targets: how AnswerGate calls the pipeline under test."""

from __future__ import annotations

from typing import Protocol

from answergate.config import Config, HttpTargetConfig
from answergate.models import TargetResponse


class Target(Protocol):
    async def ask(self, question: str) -> TargetResponse: ...

    async def aclose(self) -> None: ...


def make_target(cfg: Config) -> Target:
    if isinstance(cfg.target, HttpTargetConfig):
        from answergate.targets.http import HttpTarget

        return HttpTarget(cfg.target)
    from answergate.targets.python import PythonTarget

    return PythonTarget(cfg.target, cfg.root)
