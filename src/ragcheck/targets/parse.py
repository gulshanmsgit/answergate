"""Turn whatever a pipeline returns into a TargetResponse."""

from __future__ import annotations

from typing import Any

from ragcheck.models import Context, TargetResponse


class ResponseShapeError(Exception):
    pass


def dig(data: Any, path: str) -> Any:
    """Follow a dotted path like "data.result.answer" (list indexes allowed: "items.0")."""
    cur = data
    for part in path.split(".") if path else []:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            raise ResponseShapeError(f"Response has no field '{path}' (stopped at '{part}')")
    return cur


def to_response(
    data: Any,
    answer_path: str = "answer",
    contexts_path: str = "contexts",
    id_field: str = "id",
    text_field: str = "text",
) -> TargetResponse:
    if isinstance(data, TargetResponse):
        return data
    if isinstance(data, tuple) and len(data) == 2:
        data = {"answer": data[0], "contexts": data[1]}
        answer_path, contexts_path, id_field, text_field = "answer", "contexts", "id", "text"

    answer = dig(data, answer_path)
    if not isinstance(answer, str):
        raise ResponseShapeError(f"'{answer_path}' must be a string, got {type(answer).__name__}")

    try:
        raw_contexts = dig(data, contexts_path)
    except ResponseShapeError:
        raw_contexts = []
    if raw_contexts is None:
        raw_contexts = []
    if not isinstance(raw_contexts, list):
        raise ResponseShapeError(f"'{contexts_path}' must be a list")

    contexts: list[Context] = []
    for i, c in enumerate(raw_contexts):
        if isinstance(c, str):
            # A bare string is treated as its own id, so expected_sources can match on it
            contexts.append(Context(id=c, text=c))
        elif isinstance(c, dict):
            cid = c.get(id_field)
            text = c.get(text_field)
            contexts.append(
                Context(id=str(cid if cid is not None else i), text="" if text is None else str(text))
            )
        else:
            raise ResponseShapeError(f"Context {i} must be a string or an object")
    return TargetResponse(answer=answer, contexts=contexts)
