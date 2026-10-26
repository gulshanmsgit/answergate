"""Deterministic retrieval metrics, scored against each case's expected_sources.

No LLM needed: these alone are enough to gate a pipeline in CI.
"""

from __future__ import annotations

import math
from typing import Literal

from ragcheck.models import Context

RETRIEVAL_METRICS = ("hit_rate", "mrr", "ndcg", "context_precision", "context_recall")


SourceMatch = Literal["exact", "basename"]


def _norm(source: str, match: SourceMatch) -> str:
    """exact: compare ids as written. basename: ignore folders, case and stray spaces, so
    "docs/Refund-Policy.md" from the pipeline matches "refund-policy.md" in the golden set."""
    s = source.strip()
    if match == "basename":
        s = s.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return s


def _doc(source: str) -> str:
    # Chunk ids are often "<doc>#<chunk>"; an expected source matches the doc or the exact chunk
    return source.split("#", 1)[0]


def retrieval_scores(
    contexts: list[Context], expected_sources: list[str], k: int, match: SourceMatch = "exact"
) -> dict[str, float]:
    if not expected_sources:
        return {}
    expected = {_norm(e, match) for e in expected_sources}
    top = [_norm(c.id, match) for c in contexts[:k]]
    relevant = [cid in expected or _doc(cid) in expected for cid in top]

    first = next((i for i, hit in enumerate(relevant) if hit), None)
    found = {e for e in expected if any(cid == e or _doc(cid) == e for cid in top)}
    return {
        "hit_rate": 1.0 if first is not None else 0.0,
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
        "ndcg": _ndcg(top, expected, k),
        "context_precision": sum(relevant) / len(top) if top else 0.0,
        "context_recall": len(found) / len(expected),
    }


def _ndcg(top: list[str], expected: set[str], k: int) -> float:
    """Binary nDCG@k. Each expected source earns credit once, at the rank where it first appears,
    so five chunks of the same document can't inflate the score."""
    seen: set[str] = set()
    dcg = 0.0
    for rank, cid in enumerate(top):
        source = cid if cid in expected else _doc(cid)
        if source in expected and source not in seen:
            seen.add(source)
            dcg += 1.0 / math.log2(rank + 2)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(expected), k)))
    return dcg / ideal if ideal else 0.0


def is_expected(context_id: str, expected_sources: list[str], match: SourceMatch) -> bool:
    cid = _norm(context_id, match)
    expected = {_norm(e, match) for e in expected_sources}
    return cid in expected or _doc(cid) in expected
