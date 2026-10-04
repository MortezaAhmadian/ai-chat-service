"""PII redaction before text leaves our boundary (sent to a third-party LLM).

Concept demonstrated: offloading synchronous work with asyncio.to_thread.
- Any blocking/CPU call inside `async def` freezes the WHOLE event loop:
  every other request stalls.
- to_thread runs it in the default ThreadPoolExecutor so the loop stays
  responsive. Because of the GIL, pure-Python CPU work does not run in
  parallel; you gain responsiveness, not throughput. For heavy CPU work use
  a ProcessPoolExecutor (loop.run_in_executor) or a separate worker service.
- For tiny inputs the thread hop costs more than the work; we skip it below
  a size threshold. Measure before optimising.
"""

import asyncio
import re
from collections.abc import Sequence

from api.schemas import Message

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_OFFLOAD_THRESHOLD_CHARS = 2_000


def redact_pii(text: str) -> str:
    text = _EMAIL.sub("[EMAIL]", text)
    return _CARD.sub("[CARD_NUMBER]", text)


def _redact_all(messages: list[Message]) -> list[Message]:
    return [
        m.model_copy(update={"content": redact_pii(m.content)}) if m.role == "user" else m for m in messages
    ]


async def redact_messages(messages: Sequence[Message]) -> list[Message]:
    items = list(messages)
    if sum(len(m.content) for m in items) < _OFFLOAD_THRESHOLD_CHARS:
        return _redact_all(items)
    return await asyncio.to_thread(_redact_all, items)
