"""Deterministic in-process LLM used for local development and tests.

- With no script, it produces schema-valid JSON based on the last user message.
- A script (list of strings / exceptions) lets tests dictate exact behaviour:
  "first return garbage, then valid JSON", "fail twice with 500", etc.
- It records concurrency (in_flight / max_in_flight) so tests can PROVE that
  the semaphore actually limits parallel calls.

Magic tokens in a message (for end-to-end tests):
  __error__     -> non-retryable provider rejection
  __upstream__  -> retryable upstream 5xx
  __invalid__   -> returns prose instead of JSON
"""

import asyncio
import json
import re
from collections import deque
from collections.abc import AsyncGenerator, Iterable, Sequence
from typing import Any

from api.errors import LLMBadRequestError, LLMServerError
from api.llm.base import LLMResult
from api.schemas import Message

ScriptItem = str | BaseException

_MATH = re.compile(r"^[0-9+\-*/(). ]+$")
_WEATHER = re.compile(r"weather\s+in\s+([A-Za-z][A-Za-z .'-]{0,60})", re.IGNORECASE)
_NEGATIVE = {"bad", "hate", "angry", "terrible", "awful"}


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def default_reply(user_text: str) -> str:
    text = user_text.strip()
    if "__error__" in text:
        raise LLMBadRequestError("simulated provider rejection")
    if "__upstream__" in text:
        raise LLMServerError("simulated upstream failure")
    if "__invalid__" in text:
        return "Sorry, I can't answer in JSON today."

    action: dict[str, Any]
    if _MATH.fullmatch(text):
        action = {"type": "tool_call", "tool": "calculator", "arguments": {"expression": text}}
    elif match := _WEATHER.search(text):
        city = match.group(1).strip(" .?!")
        action = {"type": "tool_call", "tool": "get_weather", "arguments": {"city": city}}
    elif len(text.split()) < 3:
        action = {"type": "clarification", "question": "Could you tell me more about what you need?"}
    else:
        action = {"type": "answer", "text": f"(fake) You asked: {text[:200]}", "confidence": 0.42}

    words = {w.strip(".,!?").lower() for w in text.split()}
    sentiment = "negative" if words & _NEGATIVE else "neutral"
    return json.dumps({"action": action, "sentiment": sentiment, "language": "en"})


class FakeLLMClient:
    name = "fake"

    def __init__(
        self,
        script: Iterable[ScriptItem] = (),
        *,
        delay_s: float = 0.0,
        model: str = "fake-llm-1",
    ) -> None:
        self._script: deque[ScriptItem] = deque(script)
        self.delay_s = delay_s
        self.model = model
        self.calls: list[list[Message]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.closed = False

    def enqueue(self, *items: ScriptItem) -> None:
        self._script.extend(items)

    def _next_text(self, messages: Sequence[Message]) -> str:
        if self._script:
            item = self._script.popleft()
            if isinstance(item, BaseException):
                raise item
            return item
        return default_reply(messages[-1].content)

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResult:
        self.calls.append(list(messages))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            text = self._next_text(messages)
        finally:
            self.in_flight -= 1
        prompt_tokens = _approx_tokens(system) + sum(_approx_tokens(m.content) for m in messages)
        return LLMResult(
            text=text,
            model=self.model,
            input_tokens=prompt_tokens,
            output_tokens=_approx_tokens(text),
        )

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[str, None]:
        result = await self.complete(
            system=system, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        for i in range(0, len(result.text), 8):
            await asyncio.sleep(0)  # hand control back to the loop, like real network I/O
            yield result.text[i : i + 8]

    async def aclose(self) -> None:
        self.closed = True
