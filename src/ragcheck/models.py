"""Data that flows through a run: golden cases in, scored results out."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GoldenCase(BaseModel):
    """One question in the golden set (one line of the JSONL file)."""

    id: str
    question: str
    expected_sources: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    tags: list[str] = Field(default_factory=list)
    # The pipeline should decline to answer (an off-topic or unsafe question)
    expect_refusal: bool = False
    # A token planted in a poisoned document; it must never appear in the answer
    canary: str | None = None


class Context(BaseModel):
    """A chunk the pipeline retrieved."""

    id: str
    text: str = ""


class TargetResponse(BaseModel):
    answer: str
    contexts: list[Context] = Field(default_factory=list)


class CaseResult(BaseModel):
    """Everything measured for one golden case."""

    case: GoldenCase
    answer: str = ""
    contexts: list[Context] = Field(default_factory=list)
    latency_ms: float = 0.0
    # The pipeline failed for this question
    error: str | None = None
    # The judge failed, so some judged scores are missing for this question
    judge_error: str | None = None
    # Metric name -> score in [0, 1]; absent when the metric doesn't apply to this case
    scores: dict[str, float] = Field(default_factory=dict)
    # Guardrail name -> passed?
    guardrails: dict[str, bool] = Field(default_factory=dict)
    # Short human-readable explanations, shown in the report
    notes: dict[str, str] = Field(default_factory=dict)


class RunResult(BaseModel):
    """A complete run: saved to disk and compared against a baseline."""

    id: str
    created_at: str
    label: str = ""
    config_hash: str = ""
    results: list[CaseResult]
    # Metric name -> mean over the cases it applies to
    metrics: dict[str, float] = Field(default_factory=dict)
    # Guardrail name -> pass rate
    guardrails: dict[str, float] = Field(default_factory=dict)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    errors: int = 0
    judge_errors: int = 0
    source_match: Literal["exact", "basename"] = "exact"
