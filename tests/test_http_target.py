"""The HTTP target against a real local server that misbehaves in the usual ways."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from answergate.config import Config, load_golden
from answergate.models import RunResult
from answergate.runner import run
from answergate.targets import make_target

CALLS: dict[str, int] = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        CALLS[self.path] = CALLS.get(self.path, 0) + 1
        if self.path == "/ok":
            self._send(
                200,
                {
                    "result": {
                        "text": f"echo {body['q']}",
                        "docs": [{"doc_id": "refund-policy.md", "chunk": "30 days"}],
                    }
                },
            )
        elif self.path == "/html":
            self._send(502, "<html>Bad gateway</html>", "text/html")
        elif self.path == "/flaky":
            # Fails the first call for each question, then works
            if CALLS[self.path] % 2 == 1:
                self._send(503, {"error": "busy"})
            else:
                self._send(200, {"result": {"text": "ok", "docs": []}})
        elif self.path == "/slow":
            time.sleep(1.5)
            self._send(200, {"result": {"text": "late", "docs": []}})
        elif self.path == "/auth":
            ok = self.headers.get("Authorization") == "Bearer s3cret"
            self._send(200 if ok else 401, {"result": {"text": "ok", "docs": []}})

    def _send(self, code: int, payload: object, ctype: str = "application/json") -> None:
        data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def go(project: Path, url: str, **extra: object) -> RunResult:
    cfg = Config.model_validate(
        {
            "target": {
                "type": "http",
                "url": url,
                "body": {"q": "{question}", "opts": {"lang": "en"}},
                "answer_path": "result.text",
                "contexts_path": "result.docs",
                "context_id_field": "doc_id",
                "context_text_field": "chunk",
                "headers": {"Authorization": "Bearer ${RC_TOKEN}"},
            },
            "judge": {"provider": "none"},
            **extra,
        }
    )
    cfg.root = project
    target = make_target(cfg)

    async def main() -> RunResult:
        try:
            return await run(cfg, load_golden(project / "golden.jsonl"), target, None)
        finally:
            await target.aclose()

    return asyncio.run(main())


@pytest.fixture(autouse=True)
def token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RC_TOKEN", "s3cret")


def test_ok_with_nested_paths_and_custom_fields(project: Path, server: str) -> None:
    r = go(project, f"{server}/ok")
    assert r.errors == 0
    refund = next(c for c in r.results if c.case.id == "refund")
    assert refund.answer.startswith("echo How long") and refund.scores["hit_rate"] == 1.0


def test_html_error_page_is_a_case_error(project: Path, server: str) -> None:
    r = go(project, f"{server}/html", retries=0)
    assert r.errors == 4 and "502" in (r.results[0].error or "")


def test_transient_503_is_retried(project: Path, server: str) -> None:
    r = go(project, f"{server}/flaky", retries=2, concurrency=1)
    assert r.errors == 0


def test_slow_endpoint_times_out(project: Path, server: str) -> None:
    r = go(project, f"{server}/slow", retries=0, target_timeout_s=0.3)
    assert r.errors == 4 and "Timeout" in (r.results[0].error or "")


def test_secret_header_from_env(project: Path, server: str) -> None:
    assert go(project, f"{server}/auth").errors == 0
