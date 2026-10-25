from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import sys
import threading
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


async def _in_daemon_thread(fn: Callable[[str], Any], arg: str) -> Any:
    """Run a blocking call in a daemon thread.

    asyncio.to_thread would use the default executor, which asyncio.run waits for on exit: one hung
    pipeline call would then freeze the CLI long after its timeout. A daemon thread is abandoned instead.
    """
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[Any] = loop.create_future()

    def settle(ok: bool, value: Any) -> None:
        if not fut.done():  # already cancelled by a timeout
            fut.set_result(value) if ok else fut.set_exception(value)

    def work() -> None:
        try:
            result: Any = fn(arg)
            ok = True
        except BaseException as e:
            result, ok = e, False
        # RuntimeError: the run finished and its loop closed while this call was still hanging
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(settle, ok, result)

    threading.Thread(target=work, daemon=True, name="ragcheck-target").start()
    return await fut


class PythonTarget:
    """Calls a function in-process: fn(question) -> {"answer", "contexts"} or (answer, contexts)."""

    def __init__(self, cfg: PythonTargetConfig, root: Path) -> None:
        self.fn = load_callable(cfg.callable, root)

    async def ask(self, question: str) -> TargetResponse:
        if inspect.iscoroutinefunction(self.fn):
            out = await self.fn(question)
        else:
            out = await _in_daemon_thread(self.fn, question)
        # functools.partial of an async function, or an object with async __call__, returns a coroutine
        if inspect.isawaitable(out):
            out = await out
        return to_response(out)

    async def aclose(self) -> None:
        return None
