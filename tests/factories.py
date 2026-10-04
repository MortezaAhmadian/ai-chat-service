"""Test data builders (plain functions, importable; fixtures live in conftest)."""

import json
from typing import Any


def reply_json(action: dict[str, Any] | None = None, **overrides: Any) -> str:
    payload: dict[str, Any] = {
        "action": action or {"type": "answer", "text": "Paris.", "confidence": 0.9},
        "sentiment": "neutral",
        "language": "en",
    }
    payload.update(overrides)
    return json.dumps(payload)


def chat_body(content: str = "What is the capital of France?", **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"messages": [{"role": "user", "content": content}]}
    body.update(overrides)
    return body


def parse_sse(text: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for frame in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.splitlines() if ": " in line)
        events.append((fields["event"], fields["data"]))
    return events
