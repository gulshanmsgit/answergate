from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ragcheck.cli import app
from ragcheck.generate import Bm25, check_llm_question, generate_llm, generate_structure, load_sections
from ragcheck.judges import CachedJudge

CORS = """# CORS

## Use CORSMiddleware

The following arguments are supported:

* `allow_methods` - A list of HTTP methods that should be allowed. Defaults to `['GET']`. You can use `['*']`.
* `max_age` - Sets a maximum time in seconds for browsers to cache CORS responses. Defaults to `600`.

## Wildcards

It's also possible to declare the list as "*" to say that all are allowed, but not with credentials.
"""

GZIP = """# Middleware

## GZipMiddleware

Handles GZip responses for any request that includes "gzip" in the Accept-Encoding header.

* `minimum_size` - Do not GZip responses that are smaller than this minimum size in bytes. Defaults to `500`.
* `compresslevel` - Used during GZip compression. It is an integer ranging from 1 to 9. Defaults to `9`.

## Recap

You can add middleware to your application and it will run for every request and response.
"""

TESTING = """# Testing

## Using TestClient

To use TestClient, first install httpx. Notice that the testing functions are normal def, not async def.

```python
# ## Not a heading: inside a code block
def test_read_main(): ...
```
"""


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    (d / "tutorial").mkdir(parents=True)
    (d / "tutorial" / "cors.md").write_text(CORS, encoding="utf-8")
    (d / "advanced").mkdir()
    (d / "advanced" / "middleware.md").write_text(GZIP, encoding="utf-8")
    (d / "tutorial" / "testing.md").write_text(TESTING, encoding="utf-8")
    (d / "image.png").write_bytes(b"\x89PNG")
    return d


def test_sections_split_at_headings_but_not_inside_code(docs: Path) -> None:
    sections = load_sections(docs)
    assert {s.doc for s in sections} == {"tutorial/cors.md", "advanced/middleware.md", "tutorial/testing.md"}
    testing = [s for s in sections if s.doc == "tutorial/testing.md"]
    assert len(testing) == 1 and "Not a heading" in testing[0].text


def test_structure_mode_finds_defaults_with_their_source(docs: Path) -> None:
    cases = generate_structure(load_sections(docs), n=20).cases
    by_q = {c.question: c for c in cases}
    q = "What does `allow_methods` in CORS default to?"
    assert q in by_q, list(by_q)
    assert by_q[q].reference_answer == "`['GET']`" and by_q[q].expected_sources == ["tutorial/cors.md"]
    assert by_q["What does `minimum_size` in GZipMiddleware default to?"].reference_answer == "`500`"
    assert all(c.tags[:2] == ["generated", "structure"] for c in cases)
    assert not any("Recap" in c.question for c in cases)  # a vague heading isn't a question
    assert len({c.id for c in cases}) == len(cases)


def test_structure_mode_is_deterministic(docs: Path) -> None:
    a = generate_structure(load_sections(docs), n=6, seed=3).cases
    b = generate_structure(load_sections(docs), n=6, seed=3).cases
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]


def test_a_name_in_two_documents_is_skipped(docs: Path) -> None:
    (docs / "other.md").write_text(
        "# Other\n\n## Settings\n\n* `max_age` - How long sessions last in a different product altogether. Defaults to `5`.\n",
        encoding="utf-8",
    )
    questions = [c.question for c in generate_structure(load_sections(docs), n=30).cases]
    assert not any("`max_age`" in q for q in questions)


def test_bm25_finds_the_right_document(docs: Path) -> None:
    index = Bm25(load_sections(docs))
    assert index.top_docs("gzip minimum size bytes")[0] == "advanced/middleware.md"


GOOD = {
    "question": "How long do browsers cache CORS preflight responses by default?",
    "answer": "600 seconds",
    "evidence": "Sets a maximum time in seconds for browsers to cache CORS responses. Defaults to `600`.",
}


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ({**GOOD, "question": ""}, "no fact in passage"),
        ({**GOOD, "question": "What does the passage say about caching?"}, "malformed question"),
        ({**GOOD, "question": "How long is the cache"}, "malformed question"),
        ({**GOOD, "evidence": "Browsers cache responses for ten minutes."}, "evidence not in passage"),
        ({**GOOD, "answer": "300 seconds"}, "answer not grounded"),
        (
            {**GOOD, "question": "Sets a maximum time in seconds for browsers to cache CORS responses?"},
            "copies the passage",
        ),
    ],
)
def test_llm_questions_are_checked(docs: Path, reply: dict[str, Any], reason: str) -> None:
    sections = load_sections(docs)
    s = next(s for s in sections if "max_age" in s.text)
    assert check_llm_question(GOOD, s, Bm25(sections)) == (GOOD["question"], GOOD["answer"])
    assert check_llm_question(reply, s, Bm25(sections)) == reason


class FakeWriter:
    """A judge that writes one question per passage: a good one for CORS, junk for everything else."""

    name = "fake"

    async def complete(self, system: str, prompt: str, schema: Any = None, sample: int = 0) -> str:
        if "max_age" in prompt:
            return json.dumps(GOOD)
        return json.dumps({"question": "", "answer": "", "evidence": ""})

    async def aclose(self) -> None:
        pass


def test_llm_mode_keeps_only_checked_questions(docs: Path, tmp_path: Path) -> None:
    judge = CachedJudge(FakeWriter(), tmp_path / "cache", backoff_s=0.01)
    result = asyncio.run(generate_llm(load_sections(docs), n=5, judge=judge))
    assert [c.question for c in result.cases] == [GOOD["question"]]
    assert result.cases[0].expected_sources == ["tutorial/cors.md"] and result.cases[0].tags == [
        "generated",
        "llm",
    ]
    assert result.dropped["no fact in passage"] >= 1


def test_cli_writes_a_golden_set_that_ragcheck_can_load(
    docs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    r = runner.invoke(app, ["generate", str(docs), "--n", "4", "--refusals", "2"])
    assert r.exit_code == 0, r.output
    assert "Wrote 6 questions" in r.output
    rows = [
        json.loads(line)
        for line in (tmp_path / "golden.generated.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(1 for x in rows if x.get("expect_refusal")) == 2
    from ragcheck.config import load_golden

    assert len(load_golden(tmp_path / "golden.generated.jsonl")) == 6
    # Never overwrites a reviewed file by accident
    again = runner.invoke(app, ["generate", str(docs)])
    assert again.exit_code == 2 and "--force" in again.output


def test_cli_llm_mode_needs_a_judge(docs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ragcheck.yaml").write_text(
        "target: { type: python, callable: x:y }\njudge: { provider: none }\n", encoding="utf-8"
    )
    r = CliRunner().invoke(app, ["generate", str(docs), "--mode", "llm"])
    assert r.exit_code == 2 and "judge" in r.output
