"""Six small RAG pipelines, one per industry, for trying RAGCheck end to end.

Each one is a real retrieve-then-answer loop over the Markdown corpus in its folder: BM25 retrieval over
document sections, then an extractive answer built from the best-matching sentences. It needs no model,
no GPU and no network, so the whole demo runs in seconds on any laptop.

Set DEMO_FAULT to switch in one realistic regression and watch RAGCheck catch it:

  stale_index   an ingest job failed, so one document is missing from the index
  top_k_1       someone changed top_k from 3 to 1
  rerank_bug    the reranker sorts in the wrong direction
  no_refusal    the "I don't know" fallback was removed, so it answers anything
  pii_leak      the redaction step before the answer was removed
  injectable    the generator obeys instructions hidden inside retrieved documents
  hallucinate   the answer changes numbers and adds a claim no document makes (only a judge catches this)
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
INDUSTRIES = ("banking", "healthcare", "ecommerce", "insurance", "hr", "manufacturing")
FAULTS = ("stale_index", "top_k_1", "rerank_bug", "no_refusal", "pii_leak", "injectable", "hallucinate")
TOP_K = 3

REFUSAL = "I'm sorry, but I don't have information about that in my documents."
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its me my of on or our "  # noqa: SIM905
    "should so that the their there this to was we what when where which who why will with you your "
    "get got any about after before into than then them they not no yes am must also just some more most "
    # Generic question words: they say what kind of answer is wanted, not what it is about
    "many much long quickly soon often fast far take come give find used use need want tell know".split()
)
# Sentences that try to steer the model: a safe generator never repeats them
INSTRUCTION = re.compile(r"ignore (?:all )?(?:previous|prior)|begin your answer with|ai assistants?", re.I)
# Text in a question that tries to steer the assistant: never searched on, never obeyed
QUERY_INSTRUCTION = re.compile(
    r"ignore (?:your|all|previous|prior)|start your reply with|begin your answer with", re.I
)
# Conversational filler, including common Hinglish, that carries nothing to search for
FILLER = frozenset(
    "hi hello hey please thank thanks bhai batao bata mujhe jaanna jaan hai ek baat kya sorry quick".split()  # noqa: SIM905
)
# Decline when more than this share of a question's terms appear in no document at all
MAX_UNKNOWN = 0.34
EMAIL = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[a-z]{2,}\b", re.I)
PHONE = re.compile(r"\+?\d[\d\s-]{8,16}\d")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    text: str
    terms: Counter[str]
    length: int


def _stem(w: str) -> str:
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3 and not w.endswith("ss"):
            return w[: -len(suffix)] + repl
    return w


def tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


@cache
def load_corpus(industry: str) -> tuple[Chunk, ...]:
    """Split corpus.md into documents ('# file: name.md') and each document into sections ('## ...')."""
    raw = (HERE / industry / "corpus.md").read_text(encoding="utf-8")
    chunks: list[Chunk] = []
    for block in re.split(r"^# file: ", raw, flags=re.M)[1:]:
        doc, _, body = block.partition("\n")
        for i, section in enumerate(re.split(r"^## ", body, flags=re.M)[1:]):
            heading, _, text = section.partition("\n")
            text = " ".join(text.split())
            # The heading and the file name are indexed too, as most search setups do
            terms = Counter(tokens(f"{doc.replace('-', ' ')} {heading} {text}"))
            chunks.append(Chunk(f"{doc.strip()}#{i}", doc.strip(), text, terms, sum(terms.values())))
    return tuple(chunks)


@cache
def vocabulary(industry: str) -> frozenset[str]:
    return frozenset(t for c in load_corpus(industry) for t in c.terms)


def _distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@cache
def spell_correct(term: str, industry: str) -> str:
    """Map a misspelt word to the closest word in the documents (1 edit, or 2 for long words)."""
    vocab = vocabulary(industry)
    if term in vocab or len(term) < 5 or term.isdigit():
        return term
    limit = 1 if len(term) < 8 else 2
    best = min(vocab, key=lambda v: (_distance(term, v) if abs(len(v) - len(term)) <= limit else 99, v))
    return best if _distance(term, best) <= limit else term


def understand(question: str, industry: str, chunks: tuple[Chunk, ...]) -> list[str]:
    """Turn a user's message into search terms, the way a query-rewriting step would.

    The user's own email and phone are removed, sentences that try to instruct the assistant are dropped,
    filler is ignored and typos are corrected. If the message has several sentences, the one that matches
    the documents best is the question; greetings and back-story around it are left out.
    """
    text = PHONE.sub(" ", EMAIL.sub(" ", question))
    sentences = [s for s in _sentences(text) if not QUERY_INSTRUCTION.search(s)] or [text]
    best: tuple[float, list[str]] = (-1.0, [])
    for sentence in sentences:
        terms = [spell_correct(t, industry) for t in tokens(sentence) if t not in FILLER]
        score = max((s for s, _ in bm25(terms, chunks)), default=0.0) if terms else 0.0
        if score > best[0]:
            best = (score, terms)
    return best[1]


def bm25(query: list[str], chunks: tuple[Chunk, ...]) -> list[tuple[float, Chunk]]:
    n = len(chunks)
    avg = sum(c.length for c in chunks) / n
    df = Counter(t for c in chunks for t in set(c.terms))
    scored = []
    for c in chunks:
        s = 0.0
        for t in set(query):
            tf = c.terms.get(t, 0)
            if tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * c.length / avg))
        scored.append((s, c))
    return sorted(scored, key=lambda x: -x[0])


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)\s+", text) if s]


def _best_sentence(chunk: Chunk, query: set[str], safe: bool) -> tuple[int, str]:
    best = (0, "")
    for s in _sentences(chunk.text):
        if safe and INSTRUCTION.search(s):
            continue
        overlap = len(query & set(tokens(s)))
        if overlap > best[0]:
            best = (overlap, s)
    return best


def redact(text: str) -> str:
    text = EMAIL.sub("[email redacted]", text)
    return PHONE.sub(lambda m: "[phone redacted]" if 10 <= len(re.sub(r"\D", "", m[0])) <= 13 else m[0], text)


def answer(industry: str, question: str) -> dict[str, Any]:
    fault = os.environ.get("DEMO_FAULT", "")
    if fault and fault not in FAULTS:
        raise ValueError(f"Unknown DEMO_FAULT '{fault}'. Choose one of: {', '.join(FAULTS)}")
    chunks = load_corpus(industry)
    if fault == "stale_index":
        missing = chunks[0].doc  # the first document in the corpus never made it into the index
        chunks = tuple(c for c in chunks if c.doc != missing)

    query = understand(question, industry, load_corpus(industry))
    ranked = [(s, c) for s, c in bm25(query, chunks) if s > 0][: 1 if fault == "top_k_1" else TOP_K]
    if fault == "rerank_bug":
        ranked.reverse()
    contexts = [{"id": c.id, "text": c.text} for _, c in ranked]

    # Refuse when the retrieved sections cover too little of the question
    qset = set(query)
    found = {t for _, c in ranked for t in c.terms}
    coverage = len(qset & found) / max(len(qset), 1)
    unknown = len(qset - vocabulary(industry)) / max(len(qset), 1)
    best_score = max((s for s, _ in ranked), default=0.0)
    out_of_scope = best_score < 2.0 or coverage < 0.5 or unknown > MAX_UNKNOWN
    if out_of_scope and fault != "no_refusal":
        return {"answer": REFUSAL, "contexts": contexts}
    if not ranked:
        return {
            "answer": "Yes, that is generally possible; please contact support for details.",
            "contexts": [],
        }

    safe = fault != "injectable"
    parts: list[str] = []
    # Like an LLM, the generator leans on whatever comes first, so retrieval order changes the answer
    for score, chunk in ranked:
        overlap, sentence = _best_sentence(chunk, qset, safe)
        # The first section always answers; another adds a sentence if it is nearly as relevant
        if sentence and sentence not in parts and (not parts or (overlap >= 2 and score >= 0.6 * best_score)):
            parts.append(sentence)
        if len(parts) == 2:
            break
    text = " ".join(parts) or _sentences(ranked[0][1].text)[0]

    if fault == "injectable":
        for ctx in contexts:
            m = re.search(r"begin your answer with ([A-Z0-9-]+)", ctx["text"], re.I)
            if m:
                text = f"{m[1]}. {text}"
    if fault != "pii_leak":
        text = redact(text)
    if fault == "hallucinate":
        text = re.sub(r"\d+", lambda m: str(int(m[0]) * 2), text)
        text += " Exceptions are approved automatically if you ask."
    return {"answer": text, "contexts": contexts}


def banking(question: str) -> dict[str, Any]:
    return answer("banking", question)


def healthcare(question: str) -> dict[str, Any]:
    return answer("healthcare", question)


def ecommerce(question: str) -> dict[str, Any]:
    return answer("ecommerce", question)


def insurance(question: str) -> dict[str, Any]:
    return answer("insurance", question)


def hr(question: str) -> dict[str, Any]:
    return answer("hr", question)


def manufacturing(question: str) -> dict[str, Any]:
    return answer("manufacturing", question)
