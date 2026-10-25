"""Metrics that need an LLM judge. Each returns (score in [0, 1], short explanation)."""

from __future__ import annotations

from ragcheck.judges import CachedJudge
from ragcheck.models import Context

JUDGED_METRICS = ("faithfulness", "relevance", "correctness")

SYSTEM = (
    "You are a strict evaluator of a retrieval-augmented question answering system. "
    "You judge only from the material given to you, never from your own knowledge. Reply in JSON."
)

MAX_CONTEXT_CHARS = 6000


def _contexts_block(contexts: list[Context]) -> str:
    parts, used = [], 0
    for i, c in enumerate(contexts, 1):
        text = c.text[: max(0, MAX_CONTEXT_CHARS - used)]
        used += len(text)
        parts.append(f"[{i}] (id: {c.id})\n{text}")
        if used >= MAX_CONTEXT_CHARS:
            break
    return "\n\n".join(parts) or "(no context retrieved)"


async def faithfulness(judge: CachedJudge, answer: str, contexts: list[Context]) -> tuple[float, str]:
    """Share of the answer's factual claims that the retrieved context supports."""
    prompt = f"""Split the ANSWER into its individual factual claims. For each claim, decide whether the
CONTEXT explicitly supports it. Ignore filler like greetings, and ignore statements that the answer
does not know something.

CONTEXT:
{_contexts_block(contexts)}

ANSWER:
{answer}

Return {{"claims": [{{"claim": "...", "supported": true or false}}]}}"""
    data = await judge.ask_json(SYSTEM, prompt)
    claims = [c for c in data.get("claims", []) if isinstance(c, dict)]
    if not claims:
        return 1.0, "No factual claims to check"
    supported = [c for c in claims if c.get("supported") is True]
    unsupported = [str(c.get("claim", "")) for c in claims if c.get("supported") is not True]
    note = f"{len(supported)}/{len(claims)} claims supported"
    if unsupported:
        note += ". Unsupported: " + "; ".join(unsupported[:3])
    return len(supported) / len(claims), note


async def relevance(judge: CachedJudge, question: str, answer: str) -> tuple[float, str]:
    """Does the answer address what was asked? 1–5, normalised to [0, 1]."""
    prompt = f"""Rate how directly and completely the ANSWER addresses the QUESTION, from 1 to 5.
1 = off-topic or a non-answer, 3 = partly answers it, 5 = fully and directly answers it.
Do not judge factual correctness here.

QUESTION:
{question}

ANSWER:
{answer}

Return {{"score": 1-5, "reason": "one sentence"}}"""
    data = await judge.ask_json(SYSTEM, prompt)
    score = min(5.0, max(1.0, float(data.get("score", 1))))
    return (score - 1) / 4, f"{score:g}/5 — {data.get('reason', '')}"


async def correctness(judge: CachedJudge, question: str, answer: str, reference: str) -> tuple[float, str]:
    """Agreement with the reference answer, 1–5 normalised to [0, 1]."""
    prompt = f"""Compare the ANSWER with the REFERENCE answer to the QUESTION. Rate from 1 to 5 how well the
ANSWER agrees with the REFERENCE on the facts that matter. Wording does not matter. Missing key facts
or contradicting the reference lowers the score.

QUESTION:
{question}

REFERENCE:
{reference}

ANSWER:
{answer}

Return {{"score": 1-5, "reason": "one sentence"}}"""
    data = await judge.ask_json(SYSTEM, prompt)
    score = min(5.0, max(1.0, float(data.get("score", 1))))
    return (score - 1) / 4, f"{score:g}/5 — {data.get('reason', '')}"


async def is_refusal(judge: CachedJudge, question: str, answer: str) -> bool:
    prompt = f"""Did the ASSISTANT decline to answer the QUESTION, or say it cannot help with it or has no
information about it? Answering anyway, even partly, is not declining.

QUESTION:
{question}

ASSISTANT:
{answer}

Return {{"declined": true or false}}"""
    data = await judge.ask_json(SYSTEM, prompt)
    return data.get("declined") is True
