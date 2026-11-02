"""HTTP API for the RAG pipeline: POST /ask {"question": "..."} -> {"answer", "contexts"}.

uvicorn app:app --port 8000
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import rag
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel


class Question(BaseModel):
    question: str


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    n = await run_in_threadpool(rag.ingest)
    print(f"Indexed {n} chunks in {rag.collection()}")
    yield


app = FastAPI(title="FastAPI docs RAG", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    return {"collection": rag.collection(), **rag.settings()}


@app.post("/ask")
def ask(q: Question) -> dict[str, object]:
    return rag.answer(q.question)
