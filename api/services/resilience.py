"""Retries with exponential backoff + full jitter, and a circuit breaker.

Composition used in ChatService:   retry( breaker( timeout( llm.call ) ) )
- timeout innermost: every attempt has its own deadline.
- breaker inside retry: once the circuit opens, CircuitOpenError is NOT
  retryable, so retries stop immediately instead of hammering a dead upstream.
"""

import asyncio
import enum
import functools
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

from api.errors import AppError, CircuitOpenError, LLMServerError, LLMTimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: bool = True

    def compute_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """attempt is 0-based. Server-provided Retry-After wins (capped)."""
        if retry_after is not None:
            return min(retry_after, self.max_delay_s)
        ceiling = min(self.max_delay_s, self.base_delay_s * (2**attempt))
        # "Full jitter": spreads retries out so many clients that failed at the
        # same moment don't all retry at the same moment (thundering herd).
        return random.uniform(0.0, ceiling) if self.jitter else ceiling


def is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, AppError) and exc.retryable


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call fn until it succeeds, a non-retryable error occurs, or retries run out.

    `fn` is a FACTORY (zero-arg callable), not a coroutine: a coroutine object
    can only be awaited once, so each attempt needs a fresh one.
    `except Exception` deliberately does NOT catch asyncio.CancelledError
    (a BaseException since 3.8): cancellation must always propagate.
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:
            if not is_retryable(exc) or attempt >= policy.max_retries:
                raise
            delay = policy.compute_delay(attempt, getattr(exc, "retry_after", None))
            logger.warning(
                "retrying_after_error",
                extra={"attempt": attempt + 1, "delay_s": round(delay, 3), "error": exc.__class__.__name__},
            )
            attempt += 1
            await sleep(delay)


def with_retry(policy: RetryPolicy) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator form. ParamSpec keeps the wrapped function's exact signature for type checkers."""

    def decorator(fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await retry_async(lambda: fn(*args, **kwargs), policy)

        return wrapper

    return decorator


class CircuitState(enum.StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Classic three-state breaker.

    No asyncio.Lock is needed: on a single event loop, code between two
    `await`s runs atomically. All state checks/updates below happen without
    awaiting in between. (Add a lock only if a critical section must await.)
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        reset_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,  # monotonic: immune to wall-clock jumps
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        if self._state is CircuitState.OPEN and self._clock() - self._opened_at >= self.reset_timeout_s:
            return CircuitState.HALF_OPEN
        return self._state

    @staticmethod
    def _counts_as_failure(exc: BaseException) -> bool:
        # Only upstream health problems trip the breaker; a bad request from
        # us or a rate limit says nothing about the provider being down.
        return isinstance(exc, LLMServerError | LLMTimeoutError)

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        state = self.state
        if state is CircuitState.OPEN:
            raise CircuitOpenError(
                "LLM circuit is open; failing fast", details={"retry_in_s": self.reset_timeout_s}
            )
        is_probe = state is CircuitState.HALF_OPEN
        if is_probe:
            if self._probe_in_flight:
                raise CircuitOpenError("LLM circuit is half-open; probe already in flight")
            self._probe_in_flight = True

        try:
            result = await fn()
        except Exception as exc:
            if self._counts_as_failure(exc):
                self._record_failure()
            elif is_probe:
                self._state = CircuitState.HALF_OPEN  # inconclusive; allow another probe
            raise
        finally:
            # finally (not except) so that a CANCELLED probe also frees the slot.
            # Forgetting this leaves the breaker stuck half-open forever.
            if is_probe:
                self._probe_in_flight = False

        self._record_success()
        return result

    def _record_success(self) -> None:
        if self._state is not CircuitState.CLOSED:
            logger.info("circuit_closed")
        self._state = CircuitState.CLOSED
        self._failures = 0

    def _record_failure(self) -> None:
        self._failures += 1
        if self.state is CircuitState.HALF_OPEN or self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = self._clock()
            logger.error("circuit_opened", extra={"failures": self._failures})
