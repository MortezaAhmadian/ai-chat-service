"""Orchestration: request -> (cache) -> LLM (limited, timed out, retried,
circuit-broken) -> parse -> self-repair loop -> ChatResponse.

The route layer stays thin; all business logic lives here and is unit-testable
without HTTP.
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from uuid import uuid4

from pydantic import ValidationError

from api.errors import LLMTimeoutError, StructuredOutputError
from api.llm.base import LLMClient, LLMResult
from api.schemas import ChatRequest, ChatResponse, Message, StructuredReply, Usage
from api.services.cache import SingleFlightTTLCache
from api.services.concurrency import ConcurrencyLimiter
from api.services.redaction import redact_messages
from api.services.resilience import CircuitBreaker, RetryPolicy, retry_async
from api.services.structured import (
    build_repair_prompt,
    build_system_prompt,
    parse_structured,
    summarize_errors,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SSEEvent:
    event: str
    data: str

    def encode(self) -> str:
        # SSE frames: "event:" + "data:" lines, terminated by a blank line.
        # JSON-encoded data never contains raw newlines, so one data line is safe.
        return f"event: {self.event}\ndata: {self.data}\n\n"


def _clip(text: str, limit: int = 8000) -> str:
    return (text.strip() or "(empty reply)")[:limit]


class ChatService:
    def __init__(
        self,
        llm: LLMClient,
        *,
        limiter: ConcurrencyLimiter,
        breaker: CircuitBreaker,
        retry_policy: RetryPolicy,
        cache: SingleFlightTTLCache[ChatResponse],
        llm_timeout_s: float,
        stream_idle_timeout_s: float,
        max_repairs: int,
    ) -> None:
        self.llm = llm
        self.limiter = limiter
        self.breaker = breaker
        self._retry_policy = retry_policy
        self._cache = cache
        self._llm_timeout_s = llm_timeout_s
        self._stream_idle_timeout_s = stream_idle_timeout_s
        self._max_repairs = max_repairs

    # ------------------------------------------------------------------ public

    async def chat(self, req: ChatRequest) -> ChatResponse:
        if not req.cacheable:
            return await self._generate(req)
        response, outcome = await self._cache.get_or_compute(req.cache_key(), lambda: self._generate(req))
        return response.model_copy(update={"cache": outcome})

    async def stream_events(self, req: ChatRequest) -> AsyncGenerator[SSEEvent, None]:
        """Stream raw deltas, then validate the full text and emit a final event.

        Differences from chat():
        - No retries once bytes were sent: the client already saw partial output.
        - Idle timeout per chunk instead of one total deadline: a long answer
          that keeps streaming is healthy; silence is not.
        - The upstream generator is ALWAYS closed (finally: aclose) so a client
          disconnect also closes the provider connection and frees the slot.
        """
        system = build_system_prompt(StructuredReply)
        convo = await redact_messages(req.messages)
        chunks: list[str] = []
        async with self.limiter.slot():
            upstream = self.llm.stream(
                system=system, messages=convo, temperature=req.temperature, max_tokens=req.max_tokens
            )
            try:
                while True:
                    try:
                        async with asyncio.timeout(self._stream_idle_timeout_s):
                            chunk = await anext(upstream)
                    except StopAsyncIteration:
                        break
                    except TimeoutError as exc:
                        raise LLMTimeoutError("no data from LLM within idle timeout") from exc
                    chunks.append(chunk)
                    yield SSEEvent("delta", json.dumps({"text": chunk}))
            finally:
                await upstream.aclose()

        try:
            reply = parse_structured("".join(chunks), StructuredReply)
        except ValidationError as err:
            raise StructuredOutputError(
                "streamed output failed validation", details={"errors": summarize_errors(err)}
            ) from err
        yield SSEEvent("final", reply.model_dump_json())

    # ----------------------------------------------------------------- private

    async def _generate(self, req: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        system = build_system_prompt(StructuredReply)
        convo: list[Message] = await redact_messages(req.messages)
        tokens_in = tokens_out = 0
        last_error: ValidationError | None = None

        for attempt in range(self._max_repairs + 1):
            result = await self._complete_resilient(system, convo, req)
            tokens_in += result.input_tokens
            tokens_out += result.output_tokens
            try:
                reply = parse_structured(result.text, StructuredReply)
            except ValidationError as err:
                last_error = err
                logger.warning(
                    "structured_output_invalid",
                    extra={"attempt": attempt, "error_count": err.error_count()},
                )
                # Self-repair: show the model its own answer + the exact errors.
                convo = [
                    *convo,
                    Message(role="assistant", content=_clip(result.text)),
                    Message(role="user", content=build_repair_prompt(err)),
                ]
                continue

            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "chat_completed",
                extra={
                    "model": result.model,
                    "action": reply.action.type,
                    "repairs": attempt,
                    "latency_ms": latency_ms,
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_out,
                },
            )
            return ChatResponse(
                id=f"chat_{uuid4().hex}",
                model=result.model,
                reply=reply,
                usage=Usage(input_tokens=tokens_in, output_tokens=tokens_out),
                latency_ms=latency_ms,
                repair_attempts=attempt,
            )

        assert last_error is not None
        raise StructuredOutputError(
            "the model did not return valid structured output",
            details={"attempts": self._max_repairs + 1, "errors": summarize_errors(last_error)},
        )

    async def _complete_resilient(
        self, system: str, messages: Sequence[Message], req: ChatRequest
    ) -> LLMResult:
        async def one_attempt() -> LLMResult:
            # Slot is held per attempt, NOT during backoff sleeps between attempts.
            async with self.limiter.slot():
                try:
                    async with asyncio.timeout(self._llm_timeout_s):
                        return await self.llm.complete(
                            system=system,
                            messages=messages,
                            temperature=req.temperature,
                            max_tokens=req.max_tokens,
                        )
                except TimeoutError as exc:
                    raise LLMTimeoutError(f"LLM call exceeded {self._llm_timeout_s}s") from exc

        return await retry_async(lambda: self.breaker.call(one_attempt), self._retry_policy)
