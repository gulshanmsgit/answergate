from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

HERE = Path(__file__).parent

GOLDEN = [
    {
        "id": "refund",
        "question": "How long do I have to get a refund?",
        "expected_sources": ["refund-policy.md"],
        "reference_answer": "30 days from delivery.",
    },
    {"id": "shipping", "question": "Do you ship internationally?", "expected_sources": ["shipping.md"]},
    {"id": "weather", "question": "What is the weather in Paris?", "expect_refusal": True},
    {
        "id": "warranty",
        "question": "What does the warranty cover?",
        "expected_sources": ["warranty.md"],
        "canary": "RAGCHECK-CANARY-7731",
    },
]

CONFIG = """
target:
  type: python
  callable: fake_rag:answer
golden_set: golden.jsonl
judge:
  provider: none
k: 3
thresholds:
  min: { hit_rate: 0.5 }
  default_max_drop: 0.05
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A folder with a config, a golden set and the fake pipeline, used as the working directory."""
    shutil.copy(HERE / "fake_rag.py", tmp_path / "fake_rag.py")
    (tmp_path / "ragcheck.yaml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "golden.jsonl").write_text("\n".join(json.dumps(g) for g in GOLDEN), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAGCHECK_TEST_MODE", "good")
    return tmp_path


def quote_from_prompt(prompt: str) -> str:
    """A verbatim piece of the first retrieved chunk in a judge prompt: what an honest judge would cite."""
    m = re.search(r"<context>\n\[1\] \(id: [^)]*\)\n(.+)", prompt)
    return m.group(1).strip()[:60] if m else ""
