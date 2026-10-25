"""Cloud judges against local servers that speak the Anthropic and OpenAI wire formats.

The Anthropic judge goes through the real `anthropic` SDK (base_url pointed here), so this checks the
SDK integration end to end: request shape, typed errors, SDK retries and stop reasons.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from ragcheck.config import Config, load_golden
from ragcheck.judges import FatalJudgeError, make_judge
from ragcheck.models import RunResult
from ragcheck.runner import run
from ragcheck.targets import make_target

REQUESTS: list[dict[str, Any]] = []
COUNTS: dict[str, int] = {}
SEEN: dict[str, int] = {}


def verdict_for(prompt: str) -> dict[str, Any]:
    if "Split the answer" in prompt:
        return {"claims": [{"claim": "refunds take 30 days", "supported": True}]}
    if "decline" in prompt:
        return {"declined": True}
    if '"ok"' in prompt:
        return {"ok": True}
    return {"score": 4, "reason": "mostly answers it"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append(
            {"path": self.path, "body": body, "headers": {k.lower(): v for k, v in self.headers.items()}}
        )
        scenario = self.path.split("/")[2]
        COUNTS[scenario] = COUNTS.get(scenario, 0) + 1
        if self.path.startswith("/anthropic/"):
            self.anthropic(scenario, body)
        else:
            self.openai(scenario, body)

    # ── Anthropic Messages API ──
    def anthropic(self, scenario: str, body: dict[str, Any]) -> None:
        prompt = body["messages"][0]["content"]

        def message(content: list[dict[str, Any]], stop: str = "end_turn") -> None:
            self.send(
                200,
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": content,
                    "stop_reason": stop,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )

        def error(code: int, kind: str, msg: str, headers: dict[str, str] | None = None) -> None:
            self.send(code, {"type": "error", "error": {"type": kind, "message": msg}}, headers)

        text = json.dumps(verdict_for(prompt))
        n = COUNTS[scenario]
        if scenario == "ok":
            message([{"type": "text", "text": text}])
        elif scenario == "thinking":
            message(
                [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}]
            )
        elif scenario == "auth":
            error(401, "authentication_error", "invalid x-api-key")
        elif scenario == "nomodel":
            error(404, "not_found_error", f"model: {body['model']}")
        elif scenario == "badreq":
            error(400, "invalid_request_error", "temperature: Extra inputs are not permitted")
        elif scenario == "ratelimit":
            # Every other call is rate limited once; the SDK's own retry should absorb it
            if n % 2 == 1:
                error(429, "rate_limit_error", "slow down", {"retry-after": "0"})
            else:
                message([{"type": "text", "text": text}])
        elif scenario == "overloaded":
            if n % 2 == 1:
                error(529, "overloaded_error", "Overloaded")
            else:
                message([{"type": "text", "text": text}])
        elif scenario == "refusal":
            if '"ok"' in prompt:
                message([{"type": "text", "text": text}])
            else:
                message([], stop="refusal")
        elif scenario == "truncated":
            if '"ok"' in prompt:
                message([{"type": "text", "text": text}])
            else:
                message([{"type": "text", "text": '{"score": 4, "rea'}], stop="max_tokens")

    # ── OpenAI chat completions ──
    def openai(self, scenario: str, body: dict[str, Any]) -> None:
        prompt = body["messages"][1]["content"]

        def reply(content: str | None, finish: str = "stop", refusal: str | None = None) -> None:
            msg: dict[str, Any] = {"role": "assistant", "content": content}
            if refusal:
                msg["refusal"] = refusal
            self.send(200, {"choices": [{"index": 0, "message": msg, "finish_reason": finish}]})

        def error(code: int, msg: str, headers: dict[str, str] | None = None) -> None:
            self.send(code, {"error": {"message": msg, "type": "invalid_request_error"}}, headers)

        text = json.dumps(verdict_for(prompt))
        fmt = (body.get("response_format") or {}).get("type")
        if scenario == "ok":
            reply(text)
        elif scenario == "notemp":
            if "temperature" in body:
                error(400, "Unsupported value: 'temperature' does not support 0 with this model.")
            else:
                reply(text)
        elif scenario == "noschema":
            if fmt == "json_schema":
                error(400, "Invalid parameter: 'response_format' of type 'json_schema' is not supported.")
            else:
                reply("Sure! " + text)
        elif scenario == "auth":
            error(401, "Incorrect API key provided")
        elif scenario == "ratelimit":
            # The first call for each prompt is rate limited; concurrent calls can't steal each other's turn
            SEEN[prompt] = SEEN.get(prompt, 0) + 1
            if SEEN[prompt] == 1:
                error(429, "Rate limit reached", {"retry-after": "0"})
            else:
                reply(text)
        elif scenario == "refusal":
            if '"ok"' in prompt:
                reply(text)
            else:
                reply(None, refusal="I can't help with that.")

    def send(self, code: int, payload: Any, headers: dict[str, str] | None = None) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    REQUESTS.clear()
    COUNTS.clear()
    SEEN.clear()
    monkeypatch.setenv("FAKE_KEY", "sk-test-not-a-real-key")


def go(project: Path, provider: str, base_url: str, **judge: Any) -> RunResult:
    cfg = Config.model_validate(
        {
            "target": {"type": "python", "callable": "fake_rag:answer"},
            "judge": {"provider": provider, "base_url": base_url, "api_key_env": "FAKE_KEY", **judge},
        }
    )
    cfg.root = project
    target, j = make_target(cfg), make_judge(cfg)
    assert j is not None
    j.backoff_s = 0.01

    async def main() -> RunResult:
        try:
            await j.preflight()
            return await run(cfg, load_golden(project / "golden.jsonl"), target, j)
        finally:
            await target.aclose()
            await j.aclose()

    return asyncio.run(main())


# ── Anthropic ─────────────────────────────────────────────────────────


def test_anthropic_request_shape_and_scores(project: Path, server: str) -> None:
    r = go(project, "anthropic", f"{server}/anthropic/ok", model="claude-opus-5-5", effort="low")
    assert r.errors == 0 and r.judge_errors == 0
    assert r.metrics["faithfulness"] == 1.0 and r.metrics["relevance"] == 0.75
    req = REQUESTS[0]
    assert req["path"].endswith("/v1/messages")
    # The SDK owns auth and versioning: the key travels in a header, never in the body or URL
    auth = req["headers"].get("x-api-key") or req["headers"].get("authorization", "")
    assert "sk-test-not-a-real-key" in auth and "sk-test" not in json.dumps(req["body"])
    assert req["headers"]["anthropic-version"]
    b = req["body"]
    assert "temperature" not in b  # rejected by the API on current models
    assert b["output_config"]["format"]["type"] == "json_schema"
    assert b["output_config"]["effort"] == "low"
    assert b["model"] == "claude-opus-5-5"


def test_anthropic_default_model(project: Path, server: str) -> None:
    go(project, "anthropic", f"{server}/anthropic/ok")
    assert REQUESTS[0]["body"]["model"] == "claude-opus-5-5"


def test_anthropic_thinking_block_before_text(project: Path, server: str) -> None:
    assert go(project, "anthropic", f"{server}/anthropic/thinking").judge_errors == 0


@pytest.mark.parametrize(
    ("scenario", "words"), [("auth", "credentials"), ("nomodel", "no model"), ("badreq", "rejected")]
)
def test_anthropic_setup_errors_stop_at_preflight(
    project: Path, server: str, scenario: str, words: str
) -> None:
    with pytest.raises(FatalJudgeError, match=words):
        go(project, "anthropic", f"{server}/anthropic/{scenario}")
    assert COUNTS[scenario] == 1  # one call, not one per question


@pytest.mark.parametrize("scenario", ["ratelimit", "overloaded"])
def test_anthropic_429_and_529_are_retried(project: Path, server: str, scenario: str) -> None:
    r = go(project, "anthropic", f"{server}/anthropic/{scenario}")
    assert r.judge_errors == 0


@pytest.mark.parametrize("scenario", ["refusal", "truncated"])
def test_anthropic_refusal_and_max_tokens_are_per_question(project: Path, server: str, scenario: str) -> None:
    r = go(project, "anthropic", f"{server}/anthropic/{scenario}")
    assert r.judge_errors > 0 and len(r.results) == 4  # recorded, run completed
    assert "faithfulness" not in r.metrics


def test_anthropic_missing_key_env_is_fatal(
    project: Path, server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FAKE_KEY")
    with pytest.raises(FatalJudgeError, match="FAKE_KEY"):
        go(project, "anthropic", f"{server}/anthropic/ok")


# ── OpenAI-compatible ────────────────────────────────────────────────


def test_openai_ok_uses_json_schema(project: Path, server: str) -> None:
    r = go(project, "openai", f"{server}/openai/ok", model="gpt-test")
    assert r.judge_errors == 0
    assert REQUESTS[0]["body"]["response_format"]["type"] == "json_schema"
    assert REQUESTS[0]["headers"]["authorization"] == "Bearer sk-test-not-a-real-key"


def test_openai_drops_temperature_once(project: Path, server: str) -> None:
    r = go(project, "openai", f"{server}/openai/notemp", model="reasoning-model")
    assert r.judge_errors == 0
    rejected = [q for q in REQUESTS if "temperature" in q["body"]]
    assert len(rejected) == 1  # learned after the first 400, never sent again


def test_openai_falls_back_from_json_schema(project: Path, server: str) -> None:
    r = go(project, "openai", f"{server}/openai/noschema", model="local-model")
    assert r.judge_errors == 0  # json_object mode, chatter around the JSON tolerated


def test_openai_model_is_required(project: Path, server: str) -> None:
    with pytest.raises(FatalJudgeError, match=r"judge\.model"):
        go(project, "openai", f"{server}/openai/ok")


def test_openai_bad_key_is_fatal(project: Path, server: str) -> None:
    with pytest.raises(FatalJudgeError, match="API key"):
        go(project, "openai", f"{server}/openai/auth", model="m")


def test_openai_429_with_retry_after(project: Path, server: str) -> None:
    assert go(project, "openai", f"{server}/openai/ratelimit", model="m").judge_errors == 0


def test_openai_refusal_is_per_question(project: Path, server: str) -> None:
    r = go(project, "openai", f"{server}/openai/refusal", model="m")
    assert r.judge_errors > 0 and len(r.results) == 4


# ── Through the CLI ──────────────────────────────────────────────────


def test_cli_bad_key_exits_2_quickly(project: Path, server: str) -> None:
    from typer.testing import CliRunner

    from ragcheck.cli import app

    (project / "ragcheck.yaml").write_text(
        "target: {type: python, callable: 'fake_rag:answer'}\n"
        f"judge: {{provider: anthropic, base_url: '{server}/anthropic/auth', api_key_env: FAKE_KEY}}\n",
        encoding="utf-8",
    )
    r = CliRunner().invoke(app, ["run"])
    assert r.exit_code == 2 and "Judge setup problem" in r.output
    assert COUNTS["auth"] == 1
