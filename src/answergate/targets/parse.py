"""Turn whatever a pipeline returns into a TargetResponse."""

from __future__ import annotations

from typing import Any

from answergate.models import Context, TargetResponse


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


def _setting(v: Any) -> str:
    """One spelling per value, so 0 and 0.0 (or True and true) never show up as a change."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return f"{v:g}"
    if v is None:
        return "null"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_setting(x) for x in v) + "]"
    return str(v).strip()


def flatten(data: Any, prefix: str = "") -> dict[str, str]:
    """{"llm": {"model": "x", "temperature": 0}} -> {"llm.model": "x", "llm.temperature": "0"}."""
    if not isinstance(data, dict):
        return {prefix: _setting(data)} if prefix else {}
    out: dict[str, str] = {}
    for k, v in data.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        out.update(flatten(v, key) if isinstance(v, dict) else {key: _setting(v)})
    return out


def to_response(
    data: Any,
    answer_path: str = "answer",
    contexts_path: str = "contexts",
    id_field: str = "id",
    text_field: str = "text",
    pipeline_path: str = "pipeline",
    usage_path: str = "usage",
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
    # Optional: the pipeline's own settings, so a run can say what changed since the baseline
    try:
        pipeline = flatten(dig(data, pipeline_path)) if pipeline_path else {}
    except ResponseShapeError:
        pipeline = {}
    tokens_in, tokens_out = _usage(data, usage_path)
    return TargetResponse(
        answer=answer, contexts=contexts, pipeline=pipeline, tokens_in=tokens_in, tokens_out=tokens_out
    )


# Token-count spellings: ours, OpenAI's, Anthropic's and Ollama's
_IN_KEYS = ("input", "prompt_tokens", "input_tokens", "prompt_eval_count")
_OUT_KEYS = ("output", "completion_tokens", "output_tokens", "eval_count")


def _usage(data: Any, usage_path: str) -> tuple[int | None, int | None]:
    """Input and output token counts from an optional usage object; None for any that's missing or not a number."""
    try:
        usage = dig(data, usage_path) if usage_path else None
    except ResponseShapeError:
        return None, None
    if not isinstance(usage, dict):
        return None, None

    def count(keys: tuple[str, ...]) -> int | None:
        for k in keys:
            v = usage.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
                return int(v)
        return None

    return count(_IN_KEYS), count(_OUT_KEYS)
