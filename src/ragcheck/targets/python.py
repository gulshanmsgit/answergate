from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ragcheck.config import ConfigError, PythonTargetConfig
from ragcheck.models import TargetResponse
from ragcheck.targets.parse import to_response


def load_callable(spec: str, root: Path) -> Callable[..., Any]:
    if ":" not in spec:
        raise ConfigError(f"target.callable must look like 'module:function', got '{spec}'")
    module_name, attr = spec.split(":", 1)
    # Let the config's folder import its own modules, like pytest's rootdir
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        # Any error while importing the user's module is a setup problem, not a quality result
        raise ConfigError(f"Cannot import '{module_name}': {type(e).__name__}: {e}") from e
    fn: object = getattr(module, attr, None)
    if not callable(fn):
        raise ConfigError(f"'{attr}' in '{module_name}' is not callable")
    return fn


class PythonTarget:
    """Calls a function in-process: fn(question) -> {"answer", "contexts"} or (answer, contexts)."""

    def __init__(self, cfg: PythonTargetConfig, root: Path) -> None:
        self.fn = load_callable(cfg.callable, root)

    async def ask(self, question: str) -> TargetResponse:
        if inspect.iscoroutinefunction(self.fn):
            out = await self.fn(question)
        else:
            out = await asyncio.to_thread(self.fn, question)
        return to_response(out)

    async def aclose(self) -> None:
        return None
