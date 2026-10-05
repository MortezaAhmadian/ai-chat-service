"""Turning free-form LLM text into a validated Pydantic object.

Pipeline: raw text -> extract JSON -> model_validate_json -> (on failure)
build a repair prompt that feeds the exact validation errors back to the model.
"""

import functools
import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = "</think>"
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove reasoning traces from hybrid-thinking models (Qwen3/3.5, DeepSeek-R1).

    Two shapes occur: a full <think>...</think> block, or only the closing tag
    when the chat template already put <think> in the prompt. Reasoning often
    contains braces, which would confuse JSON extraction, so strip it first.
    """
    if _THINK_CLOSE in text.lower():
        text = text[text.lower().rindex(_THINK_CLOSE) + len(_THINK_CLOSE) :]
    return _THINK_BLOCK.sub("", text)


def extract_json_block(text: str) -> str:
    """Models often wrap JSON in ```json fences or add a sentence before it."""
    candidate = strip_reasoning(text).strip()
    if match := _FENCE.match(candidate):
        candidate = match.group(1).strip()
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]
    return candidate


def parse_structured(text: str, model: type[M]) -> M:
    """model_validate_json parses + validates in one pass in Rust (pydantic-core):
    faster than json.loads + model_validate, and malformed JSON surfaces as a
    regular ValidationError (type 'json_invalid')."""
    return model.model_validate_json(extract_json_block(text))


def summarize_errors(err: ValidationError, limit: int = 8) -> list[dict[str, str]]:
    return [
        {"loc": ".".join(str(p) for p in e["loc"]) or "<root>", "msg": e["msg"]}
        for e in err.errors(include_url=False)[:limit]
    ]


def build_repair_prompt(err: ValidationError) -> str:
    lines = "\n".join(f"- {e['loc']}: {e['msg']}" for e in summarize_errors(err))
    return (
        "Your previous reply did not match the required JSON schema.\n"
        f"Validation errors:\n{lines}\n"
        "Reply again with ONLY the corrected JSON object. No prose, no code fences."
    )


@functools.cache
def response_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema handed to providers that support constrained decoding.
    Cached: generated once per model class. Treat the returned dict as read-only."""
    return model.model_json_schema()


@functools.cache  # schema generation is not free; do it once per model
def build_system_prompt(model: type[BaseModel]) -> str:
    schema = json.dumps(model.model_json_schema(), separators=(",", ":"))
    return (
        "You are a helpful assistant inside an agent runtime.\n"
        "Decide ONE action: answer directly, call a tool, or ask a clarifying question.\n"
        "Use the calculator tool for arithmetic and get_weather for weather questions.\n"
        "Respond with a single JSON object that validates against this JSON Schema, "
        "and nothing else:\n"
        f"{schema}"
    )
