"""`answergate.yaml`: what to call, what to ask, and when to fail."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from answergate.models import GoldenCase


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
    # Optional object in the response with the pipeline's settings (model, temperature, chunk size...);
    # AnswerGate records them and reports which ones changed since the baseline
    pipeline_path: str = "pipeline"
    timeout_s: float = 60.0


class PythonTargetConfig(BaseModel):
    type: Literal["python"]
    # "package.module:function" — called with the question, sync or async
    callable: str


class JudgeConfig(BaseModel):
    provider: Literal["ollama", "openai", "anthropic", "none"] = "ollama"
    # Left out: qwen2.5:7b for ollama, claude-opus-5-5 for anthropic; openai needs it set
    model: str | None = None
    base_url: str | None = None
    # Claude only: how hard the judge thinks (low | medium | high | xhigh | max); left out = model default
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    # Name of the environment variable holding the API key (never the key itself)
    api_key_env: str | None = None
    timeout_s: float = 120.0
    # Parallel judge calls. Kept low: a local Ollama serves one request at a time, and queued calls time out
    concurrency: int = Field(default=2, ge=1)
    # Grade each metric this many times and keep the median / majority. Odd, so votes can't tie.
    samples: Literal[1, 3, 5, 7] = 1
    # Faithfulness: a "supported" verdict counts only if the judge's quote is found in the retrieved text
    verify_evidence: bool = True


# Every metric AnswerGate can compute; names in `metrics` and `thresholds` are checked against these
METRICS = (
    "hit_rate",
    "mrr",
    "ndcg",
    "context_precision",
    "context_recall",
    "faithfulness",
    "relevance",
    "correctness",
    "consistency",
)


class Thresholds(BaseModel):
    # Absolute floors: fail if a metric's mean is below this
    min: dict[str, float] = Field(default_factory=dict)
    # Regression gate: fail if a metric drops more than this versus the baseline
    max_drop: dict[str, float] = Field(default_factory=dict)
    # Used for any metric not listed in max_drop; null turns the default gate off
    default_max_drop: float | None = 0.05
    # Every guardrail case must pass
    guardrails_must_pass: bool = True
    # "all": any guardrail failure fails the gate. "new": a question that already failed the same guardrail
    # in the baseline is a known failure and only warns, so a team can adopt AnswerGate on an imperfect
    # pipeline and still block every new failure
    guardrail_gate: Literal["all", "new"] = "all"
    # Fail on a drop only when its 95% confidence interval excludes zero (fewer false alarms from noise)
    require_significance: bool = False


class Config(BaseModel):
    target: HttpTargetConfig | PythonTargetConfig = Field(discriminator="type")
    golden_set: str = "golden.jsonl"
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    metrics: list[str] = Field(
        default_factory=lambda: [
            "hit_rate",
            "mrr",
            "ndcg",
            "context_precision",
            "context_recall",
            "faithfulness",
            "relevance",
            "correctness",
            "consistency",
        ]
    )
    # Ask each question this many times; with 2+, `consistency` scores how much the answers agree
    # (catches a temperature or sampling change that makes answers unstable)
    repeats: int = Field(default=1, ge=1, le=10)
    # Top-k retrieved contexts considered by the retrieval metrics
    k: int = Field(default=5, ge=1)
    concurrency: int = Field(default=4, ge=1)
    # Extra attempts after a pipeline call fails or times out
    retries: int = Field(default=2, ge=0)
    # A pipeline call taking longer than this counts as an error for that question
    target_timeout_s: float = Field(default=120.0, gt=0)
    # Cap on how many questions per second are sent to the pipeline (None: no cap)
    rate_limit_rps: float | None = Field(default=None, gt=0)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    # How expected_sources match retrieved ids: "basename" ignores folders, case and spaces; "exact" doesn't
    source_match: Literal["basename", "exact"] = "basename"
    # Strings the PII check ignores, e.g. your public support email
    pii_allow: list[str] = Field(default_factory=list)
    # Where run history, the baseline and the judge cache live
    workdir: str = ".answergate"

    # Set by load_config: the folder relative paths resolve against
    root: Path = Field(default=Path("."), exclude=True)

    @model_validator(mode="after")
    def _known_metrics(self) -> Config:
        # A typo here would otherwise switch a metric or a gate off without a word
        for where, names in (
            ("metrics", self.metrics),
            ("thresholds.min", list(self.thresholds.min)),
            ("thresholds.max_drop", list(self.thresholds.max_drop)),
        ):
            for name in names:
                if name not in METRICS:
                    close = [m for m in METRICS if m[:4] == name[:4]]
                    hint = f" Did you mean '{close[0]}'?" if close else ""
                    raise ValueError(f"Unknown metric '{name}' in {where}.{hint} Known: {', '.join(METRICS)}")
        return self

    def path(self, p: str) -> Path:
        return (self.root / p).resolve()

    def fingerprint(self) -> str:
        """Hash of the settings that change scores, stored with each run."""
        data = self.model_dump(
            mode="json", exclude={"root", "concurrency", "retries", "thresholds", "workdir", "rate_limit_rps"}
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
        raise ConfigError(f"Config not found: {p}. Run `answergate init` to create one.")
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
    # utf-8-sig: Windows editors often save a byte-order mark that breaks the first line
    text = path.read_text(encoding="utf-8-sig")
    cases: list[GoldenCase] = []
    if text.lstrip().startswith("["):
        # A plain JSON array is accepted too
        try:
            rows = json.loads(text)
        except json.JSONDecodeError as e:
            raise ConfigError(f"{path.name} is not valid JSON: {e}") from e
        for n, row in enumerate(rows, 1):
            try:
                cases.append(GoldenCase.model_validate(row))
            except ValidationError as e:
                raise ConfigError(f"{path.name} item {n}: {e}") from e
    else:
        for n, line in enumerate(text.splitlines(), 1):
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
