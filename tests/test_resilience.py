import asyncio
import random

import pytest

from api.errors import (
    CircuitOpenError,
    LLMBadRequestError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from api.services.resilience import CircuitBreaker, CircuitState, RetryPolicy, retry_async, with_retry


class Flaky:
    """Callable factory that fails N times, then succeeds."""

    def __init__(self, failures: list[Exception], result: str = "ok") -> None:
        self.failures = list(failures)
        self.result = result
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.result


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def fake_sleep(sleeps: list[float]):  # type: ignore[no-untyped-def]
    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    return _sleep


class TestRetry:
    async def test_succeeds_after_transient_errors(self, fake_sleep, sleeps) -> None:  # type: ignore[no-untyped-def]
        fn = Flaky([LLMServerError("500"), LLMTimeoutError("slow")])
        policy = RetryPolicy(max_retries=3, base_delay_s=1, max_delay_s=10, jitter=False)
        assert await retry_async(fn, policy, sleep=fake_sleep) == "ok"
        assert fn.calls == 3
        assert sleeps == [1, 2]  # exponential: 1 * 2**0, 1 * 2**1

    async def test_non_retryable_error_is_raised_immediately(self, fake_sleep, sleeps) -> None:  # type: ignore[no-untyped-def]
        fn = Flaky([LLMBadRequestError("400")])
        with pytest.raises(LLMBadRequestError):
            await retry_async(fn, RetryPolicy(max_retries=5), sleep=fake_sleep)
        assert fn.calls == 1
        assert sleeps == []

    async def test_gives_up_after_max_retries(self, fake_sleep) -> None:  # type: ignore[no-untyped-def]
        fn = Flaky([LLMServerError("500")] * 10)
        with pytest.raises(LLMServerError):
            await retry_async(fn, RetryPolicy(max_retries=2), sleep=fake_sleep)
        assert fn.calls == 3  # 1 attempt + 2 retries

    async def test_retry_after_header_wins_but_is_capped(self, fake_sleep, sleeps) -> None:  # type: ignore[no-untyped-def]
        fn = Flaky([LLMRateLimitError("429", retry_after=3.0), LLMRateLimitError("429", retry_after=60.0)])
        await retry_async(fn, RetryPolicy(max_retries=3, max_delay_s=10), sleep=fake_sleep)
        assert sleeps == [3.0, 10.0]

    def test_full_jitter_stays_within_bounds(self) -> None:
        random.seed(1234)
        policy = RetryPolicy(base_delay_s=1, max_delay_s=8)
        delays = [policy.compute_delay(attempt=3) for _ in range(500)]
        assert all(0 <= d <= 8 for d in delays)
        assert max(delays) > 6  # actually spread out...
        assert min(delays) < 2

    async def test_cancellation_is_never_swallowed(self) -> None:
        async def hang() -> str:
            await asyncio.sleep(10)
            return "never"

        task = asyncio.create_task(retry_async(hang, RetryPolicy(max_retries=5)))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_decorator_preserves_metadata(self) -> None:
        @with_retry(RetryPolicy(max_retries=1, base_delay_s=0))
        async def fetch(x: int) -> int:
            """Docstring survives."""
            return x * 2

        assert await fetch(21) == 42
        assert fetch.__name__ == "fetch"
        assert fetch.__doc__ == "Docstring survives."


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def _fail() -> str:
    raise LLMServerError("boom")


async def _ok() -> str:
    return "ok"


class TestCircuitBreaker:
    @pytest.fixture
    def clock(self) -> FakeClock:
        return FakeClock()

    @pytest.fixture
    def breaker(self, clock: FakeClock) -> CircuitBreaker:
        return CircuitBreaker(failure_threshold=2, reset_timeout_s=10, clock=clock)

    async def _trip(self, breaker: CircuitBreaker) -> None:
        for _ in range(breaker.failure_threshold):
            with pytest.raises(LLMServerError):
                await breaker.call(_fail)

    async def test_opens_after_threshold_and_fails_fast(self, breaker: CircuitBreaker) -> None:
        await self._trip(breaker)
        assert breaker.state is CircuitState.OPEN
        calls = 0

        async def counted() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        with pytest.raises(CircuitOpenError):
            await breaker.call(counted)
        assert calls == 0  # upstream was not touched

    async def test_half_open_success_closes(self, breaker: CircuitBreaker, clock: FakeClock) -> None:
        await self._trip(breaker)
        clock.now = 10
        assert breaker.state is CircuitState.HALF_OPEN
        assert await breaker.call(_ok) == "ok"
        assert breaker.state is CircuitState.CLOSED

    async def test_half_open_failure_reopens(self, breaker: CircuitBreaker, clock: FakeClock) -> None:
        await self._trip(breaker)
        clock.now = 10
        with pytest.raises(LLMServerError):
            await breaker.call(_fail)
        assert breaker.state is CircuitState.OPEN

    async def test_client_errors_do_not_trip(self, breaker: CircuitBreaker) -> None:
        async def bad_request() -> str:
            raise LLMBadRequestError("400")

        for _ in range(5):
            with pytest.raises(LLMBadRequestError):
                await breaker.call(bad_request)
        assert breaker.state is CircuitState.CLOSED

    async def test_only_one_probe_at_a_time(self, breaker: CircuitBreaker, clock: FakeClock) -> None:
        await self._trip(breaker)
        clock.now = 10
        gate = asyncio.Event()

        async def slow_probe() -> str:
            await gate.wait()
            return "ok"

        probe = asyncio.create_task(breaker.call(slow_probe))
        await asyncio.sleep(0)
        with pytest.raises(CircuitOpenError, match="probe already in flight"):
            await breaker.call(_ok)
        gate.set()
        assert await probe == "ok"

    async def test_cancelled_probe_releases_the_slot(self, breaker: CircuitBreaker, clock: FakeClock) -> None:
        await self._trip(breaker)
        clock.now = 10

        async def hang() -> str:
            await asyncio.sleep(100)
            return "never"

        probe = asyncio.create_task(breaker.call(hang))
        await asyncio.sleep(0)
        probe.cancel()
        with pytest.raises(asyncio.CancelledError):
            await probe
        assert await breaker.call(_ok) == "ok"  # would raise if the slot leaked
