"""Edge cases for the deterministic retrieval metrics: the inputs real pipelines send that the happy path misses."""

from __future__ import annotations

import math
import random

import pytest

from answergate.metrics.retrieval import RETRIEVAL_METRICS, is_expected, retrieval_scores
from answergate.models import Context


def ctx(*ids: str) -> list[Context]:
    return [Context(id=i) for i in ids]


def test_nothing_retrieved_scores_zero_everywhere() -> None:
    # A retriever that returns nothing (index empty, filter too strict) must fail every metric, not crash
    assert retrieval_scores([], ["a.md"], k=5) == dict.fromkeys(RETRIEVAL_METRICS, 0.0)


def test_a_duplicated_expected_source_counts_once() -> None:
    # A golden set that lists the same page twice must not halve recall
    s = retrieval_scores(ctx("a.md#0"), ["a.md", "a.md"], k=5)
    assert s["context_recall"] == 1.0
    assert s["ndcg"] == 1.0


def test_an_expected_chunk_is_not_satisfied_by_a_neighbouring_chunk() -> None:
    # Expecting chunk 2 exactly: chunk 1 of the same page is a different passage and earns nothing
    s = retrieval_scores(ctx("a.md#1", "b.md#0"), ["a.md#2"], k=5)
    assert s["hit_rate"] == 0.0 and s["context_recall"] == 0.0


def test_chunk_ids_with_more_than_one_hash_belong_to_their_page() -> None:
    # "<doc>#<section>#<chunk>" still belongs to <doc>
    s = retrieval_scores(ctx("guide.md#install#3"), ["guide.md"], k=5)
    assert s["hit_rate"] == 1.0 and s["mrr"] == 1.0


def test_ndcg_ideal_is_capped_at_k_when_more_sources_are_expected() -> None:
    # Three relevant pages but only two slots: filling both slots with relevant pages is a perfect ranking
    s = retrieval_scores(ctx("a.md", "b.md", "c.md"), ["a.md", "b.md", "c.md"], k=2)
    assert s["ndcg"] == pytest.approx(1.0)
    assert s["context_recall"] == pytest.approx(2 / 3)


def test_precision_and_recall_move_independently() -> None:
    # One right page in four results: precision falls, recall stays full
    s = retrieval_scores(ctx("x", "a.md", "y", "z"), ["a.md"], k=4)
    assert s["context_precision"] == 0.25
    assert s["context_recall"] == 1.0
    assert s["mrr"] == 0.5
    assert s["ndcg"] == pytest.approx(1 / math.log2(3))


def test_exact_mode_ignores_stray_whitespace_but_not_case() -> None:
    assert retrieval_scores(ctx(" a.md "), ["a.md"], k=5)["hit_rate"] == 1.0
    assert retrieval_scores(ctx("A.md"), ["a.md"], k=5)["hit_rate"] == 0.0


@pytest.mark.parametrize(
    "returned",
    [
        "C:\\docs\\Refund-Policy.md",
        "docs/refund-policy.md#4",
        "  REFUND-POLICY.MD  ",
        "s3://bucket/x/refund-policy.md",
    ],
)
def test_basename_mode_matches_paths_from_any_system(returned: str) -> None:
    assert retrieval_scores(ctx(returned), ["refund-policy.md"], k=3, match="basename")["hit_rate"] == 1.0
    assert is_expected(returned, ["refund-policy.md"], "basename")


def test_is_expected_agrees_with_the_metrics() -> None:
    # The HTML report highlights passages with is_expected; it must mark exactly the ones the metrics count
    ids = ["a.md#0", "b.md", "a.md#7", "c.md#1"]
    expected = ["a.md", "c.md#1"]
    marked = [is_expected(i, expected, "exact") for i in ids]
    assert marked == [True, False, True, True]
    assert retrieval_scores(ctx(*ids), expected, k=4)["context_precision"] == sum(marked) / len(ids)


def test_scores_always_stay_between_zero_and_one() -> None:
    # Seeded random rankings, duplicates and chunk ids included: no metric may leave [0, 1]
    rng = random.Random(7)
    pool = ["a.md", "b.md", "c.md", "d.md", "a.md#1", "b.md#2", "x", "y"]
    for _ in range(500):
        ranking = rng.choices(pool, k=rng.randint(0, 8))
        expected = rng.sample(["a.md", "b.md", "c.md", "a.md#1"], k=rng.randint(1, 3))
        scores = retrieval_scores(ctx(*ranking), expected, k=rng.randint(1, 6))
        assert set(scores) == set(RETRIEVAL_METRICS)
        assert all(0.0 <= v <= 1.0 for v in scores.values()), (ranking, expected, scores)
