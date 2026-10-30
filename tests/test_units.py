from __future__ import annotations

import math

import pytest

from ragcheck.baseline import evaluate
from ragcheck.config import Thresholds
from ragcheck.guardrails import find_pii, leaked_canary, looks_like_refusal
from ragcheck.judges import JudgeError, parse_json
from ragcheck.metrics.retrieval import retrieval_scores
from ragcheck.models import CaseResult, Context, GoldenCase, RunResult
from ragcheck.report import markdown
from ragcheck.targets.parse import ResponseShapeError, to_response


def ctx(*ids: str) -> list[Context]:
    return [Context(id=i, text="") for i in ids]


class TestRetrieval:
    def test_first_hit(self) -> None:
        s = retrieval_scores(ctx("a.md#0", "b.md#1"), ["a.md"], k=5)
        assert s == {
            "hit_rate": 1.0,
            "mrr": 1.0,
            "ndcg": 1.0,
            "context_precision": 0.5,
            "context_recall": 1.0,
        }

    def test_ndcg_rewards_rank_and_counts_each_source_once(self) -> None:
        # Second expected doc found at rank 3: DCG = 1 + 1/log2(4) = 1.5, ideal = 1 + 1/log2(3)
        s = retrieval_scores(ctx("a.md#0", "a.md#1", "b.md"), ["a.md", "b.md"], k=5)
        assert s["ndcg"] == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
        # Repeating the same document doesn't add credit
        assert retrieval_scores(ctx("a.md#0", "a.md#1", "a.md#2"), ["a.md"], k=5)["ndcg"] == 1.0
        assert retrieval_scores(ctx("x", "a.md"), ["a.md"], k=5)["ndcg"] == pytest.approx(1 / math.log2(3))
        assert retrieval_scores(ctx("x", "y"), ["a.md"], k=5)["ndcg"] == 0.0

    def test_second_hit_and_recall(self) -> None:
        s = retrieval_scores(ctx("x", "b.md#3", "y"), ["b.md", "c.md"], k=5)
        assert s["mrr"] == 0.5
        assert s["context_recall"] == 0.5
        assert s["context_precision"] == pytest.approx(1 / 3)

    def test_miss_beyond_k(self) -> None:
        s = retrieval_scores(ctx("x", "y", "a.md"), ["a.md"], k=2)
        assert s["hit_rate"] == 0.0 and s["mrr"] == 0.0

    def test_no_expected_sources_means_not_applicable(self) -> None:
        assert retrieval_scores(ctx("a"), [], k=5) == {}

    def test_exact_chunk_id(self) -> None:
        assert retrieval_scores(ctx("a.md#1", "a.md#2"), ["a.md#2"], k=5)["mrr"] == 0.5


class TestGuardrails:
    @pytest.mark.parametrize(
        ("text", "kind"),
        [
            ("Write to jane.doe@corp.com", "email"),
            ("Call +91 98450 12345 now", "phone"),
            ("Aadhaar 2345 6789 0123", "aadhaar"),
            ("PAN ABCDE1234F", "pan"),
            ("card 4111 1111 1111 1111", "card"),
            ("ssn 123-45-6789", "us_ssn"),
        ],
    )
    def test_detects(self, text: str, kind: str) -> None:
        assert kind in find_pii(text)

    def test_clean_text_and_numbers(self) -> None:
        assert find_pii("Refunds within 30 days. We ship to 40 countries since 2019.") == []

    def test_ip_addresses_and_ids_are_not_phones(self) -> None:
        # Both seen in real answers about the FastAPI docs
        assert find_pii('fastapi run --forwarded-allow-ips="192.168.1.100"') == []
        assert find_pii('"Client #1596980209979 left the chat"') == []

    def test_card_needs_valid_checksum(self) -> None:
        assert "card" not in find_pii("order 1234 5678 9012 3456")

    def test_allow_list(self) -> None:
        assert find_pii("Mail support@shop.com", allow=["support@shop.com"]) == []

    def test_canary(self) -> None:
        assert leaked_canary("ragcheck-canary-7731 hi", "RAGCHECK-CANARY-7731")
        assert not leaked_canary("one-year warranty", "RAGCHECK-CANARY-7731")

    def test_refusal_heuristic(self) -> None:
        assert looks_like_refusal("I'm sorry, but I can't help with that.")
        assert looks_like_refusal("That is not in the documents I have.")
        assert looks_like_refusal("I don't know based on the FastAPI docs.")
        assert not looks_like_refusal("Refunds are accepted within 30 days.")


class TestParsing:
    def test_dotted_paths_and_dict_contexts(self) -> None:
        r = to_response(
            {"data": {"out": "hi", "src": [{"doc": "a", "body": "t"}]}}, "data.out", "data.src", "doc", "body"
        )
        assert r.answer == "hi" and r.contexts[0].id == "a" and r.contexts[0].text == "t"

    def test_string_contexts_and_tuple(self) -> None:
        r = to_response(("ans", ["chunk one"]))
        assert r.contexts[0].id == "chunk one"

    def test_missing_answer(self) -> None:
        with pytest.raises(ResponseShapeError):
            to_response({"text": "x"})

    def test_judge_json_with_fences(self) -> None:
        assert parse_json('Sure!\n```json\n{"score": 4}\n```') == {"score": 4}
        with pytest.raises(JudgeError):
            parse_json("no json here")


def make_run(
    rid: str, scores: dict[str, dict[str, float]], rails: dict[str, dict[str, bool]] | None = None
) -> RunResult:
    from ragcheck.runner import aggregate

    results = [
        CaseResult(
            case=GoldenCase(id=cid, question=f"q {cid}"), scores=s, guardrails=(rails or {}).get(cid, {})
        )
        for cid, s in scores.items()
    ]
    m, g = aggregate(results)
    return RunResult(id=rid, created_at="now", results=results, metrics=m, guardrails=g)


class TestGate:
    def test_passes_without_baseline(self) -> None:
        v = evaluate(make_run("a", {"1": {"hit_rate": 1.0}}), None, Thresholds())
        assert v.passed

    def test_minimum(self) -> None:
        v = evaluate(make_run("a", {"1": {"hit_rate": 0.5}}), None, Thresholds(min={"hit_rate": 0.8}))
        assert not v.passed and "below the minimum" in v.failures[0]

    def test_drop_beyond_allowed_fails_and_lists_question(self) -> None:
        base = make_run("b", {"1": {"hit_rate": 1.0}, "2": {"hit_rate": 1.0}})
        cur = make_run("c", {"1": {"hit_rate": 0.0}, "2": {"hit_rate": 1.0}})
        v = evaluate(cur, base, Thresholds(default_max_drop=0.05))
        assert not v.passed
        assert [r.case_id for r in v.regressions] == ["1"]

    def test_small_drop_within_allowance(self) -> None:
        base = make_run("b", {"1": {"faithfulness": 0.90}})
        cur = make_run("c", {"1": {"faithfulness": 0.88}})
        assert evaluate(cur, base, Thresholds(default_max_drop=0.05)).passed

    def test_per_metric_override(self) -> None:
        base = make_run("b", {"1": {"relevance": 0.9}})
        cur = make_run("c", {"1": {"relevance": 0.82}})
        assert evaluate(cur, base, Thresholds(max_drop={"relevance": 0.1})).passed

    def test_guardrail_failure(self) -> None:
        cur = make_run("c", {"1": {}}, {"1": {"pii": False}})
        v = evaluate(cur, None, Thresholds())
        assert not v.passed and "pii" in v.failures[0]

    def test_markdown_summary(self) -> None:
        base = make_run("b", {"1": {"hit_rate": 1.0}})
        cur = make_run("c", {"1": {"hit_rate": 0.0}})
        md = markdown(cur, evaluate(cur, base, Thresholds()))
        assert "RAGCheck failed" in md and "| Hit rate | 1.000 | 0.000 | −1.000 [−1.000, −1.000] |" in md
