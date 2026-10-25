"""Deterministic retrieval metrics, scored against each case's expected_sources.

No LLM needed: these alone are enough to gate a pipeline in CI.
"""

from __future__ import annotations

from ragcheck.models import Context

RETRIEVAL_METRICS = ("hit_rate", "mrr", "context_precision", "context_recall")


def _matches(context_id: str, expected: set[str]) -> bool:
    # Chunk ids are often "<doc>#<chunk>"; an expected source matches the doc or the exact chunk
    return context_id in expected or context_id.split("#", 1)[0] in expected


def retrieval_scores(contexts: list[Context], expected_sources: list[str], k: int) -> dict[str, float]:
    if not expected_sources:
        return {}
    expected = set(expected_sources)
    top = contexts[:k]
    relevant = [_matches(c.id, expected) for c in top]

    first = next((i for i, hit in enumerate(relevant) if hit), None)
    found = {e for e in expected if any(c.id == e or c.id.split("#", 1)[0] == e for c in top)}
    return {
        "hit_rate": 1.0 if first is not None else 0.0,
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
        "context_precision": sum(relevant) / len(top) if top else 0.0,
        "context_recall": len(found) / len(expected),
    }
