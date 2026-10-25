"""`ragcheck.yaml`: what to call, what to ask, and when to fail."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from ragcheck.models import GoldenCase


class ConfigError(Exception):
    pass


class HttpTargetConfig(BaseModel):
    type: Literal["http"]
    url: str
    method: Literal["POST", "GET"] = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    # Request body; the string "{question}" anywhere in it is replaced with the question
    body: dict[str, object] = Field(default_factory=lambda: dict[str, object](question="{question}"))
    # Dotted paths into the JSON response
    answer_path: str = "answer"
    contexts_path: str = "contexts"
    context_id_field: str = "id"
    context_text_field: str = "text"
    timeout_s: float = 60.0


class PythonTargetConfig(BaseModel):
    type: Literal["python"]
    # "package.module:function" — called with the question, sync or async
    callable: str


class JudgeConfig(BaseModel):
    provider: Literal["ollama", "openai", "anthropic", "none"] = "ollama"
    model: str = "qwen2.5:7b"
    base_url: str | None = None
    # Name of the environment variable holding the API key (never the key itself)
    api_key_env: str | None = None
    timeout_s: float = 120.0


class Thresholds(BaseModel):
    # Absolute floors: fail if a metric's mean is below this
    min: dict[str, float] = Field(default_factory=dict)
    # Regression gate: fail if a metric drops more than this versus the baseline
    max_drop: dict[str, float] = Field(default_factory=dict)
    # Used for any metric not listed in max_drop; null turns the default gate off
    default_max_drop: float | None = 0.05
    # Every guardrail case must pass
    guardrails_must_pass: bool = True


class Config(BaseModel):
    target: HttpTargetConfig | PythonTargetConfig = Field(discriminator="type")
    golden_set: str = "golden.jsonl"
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    metrics: list[str] = Field(
        default_factory=lambda: [
            "hit_rate",
            "mrr",
            "context_precision",
            "faithfulness",
            "relevance",
            "correctness",
        ]
    )
    # Top-k retrieved contexts considered by the retrieval metrics
    k: int = 5
    concurrency: int = 4
    retries: int = 2
    thresholds: Thresholds = Field(default_factory=Thresholds)
    # Strings the PII check ignores, e.g. your public support email
    pii_allow: list[str] = Field(default_factory=list)
    # Where run history, the baseline and the judge cache live
    workdir: str = ".ragcheck"

    # Set by load_config: the folder relative paths resolve against
    root: Path = Field(default=Path("."), exclude=True)

    def path(self, p: str) -> Path:
        return (self.root / p).resolve()

    def fingerprint(self) -> str:
        """Hash of the settings that change scores, stored with each run."""
        data = self.model_dump(
            mode="json", exclude={"root", "concurrency", "retries", "thresholds", "workdir"}
        )
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: str) -> str:
    """Replace ${VAR} with its environment value, so secrets stay out of the YAML."""

    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in os.environ:
            raise ConfigError(f"Environment variable {name} is not set")
        return os.environ[name]

    return _ENV.sub(sub, value)


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config not found: {p}. Run `ragcheck init` to create one.")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        cfg = Config.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as e:
        raise ConfigError(f"Invalid config {p}:\n{e}") from e
    cfg.root = p.parent.resolve()
    return cfg


def load_golden(path: Path) -> list[GoldenCase]:
    if not path.exists():
        raise ConfigError(f"Golden set not found: {path}")
    cases: list[GoldenCase] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            cases.append(GoldenCase.model_validate_json(line))
        except ValidationError as e:
            raise ConfigError(f"{path.name} line {n}: {e}") from e
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ConfigError(f"Duplicate case ids in {path.name}: {', '.join(sorted(dupes))}")
    if not cases:
        raise ConfigError(f"Golden set {path.name} is empty")
    return cases
