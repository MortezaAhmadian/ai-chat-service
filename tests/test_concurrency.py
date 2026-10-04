import asyncio
import contextvars
import time

import pytest

from api.errors import LLMBadRequestError, OverloadedError
from api.services.concurrency import ConcurrencyLimiter, gather_bounded, run_all_or_raise


async def test_limiter_caps_parallelism() -> None:
    limiter = ConcurrencyLimiter(3, acquire_timeout_s=5)
    active = peak = 0

    async def job() -> None:
        nonlocal active, peak
        async with limiter.slot():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*(job() for _ in range(12)))
    assert peak == 3
    assert limiter.in_use == 0


async def test_limiter_sheds_load_when_saturated() -> None:
    limiter = ConcurrencyLimiter(1, acquire_timeout_s=0.02)
    release = asyncio.Event()

    async def holder() -> None:
        async with limiter.slot():
            await release.wait()

    task = asyncio.create_task(holder())
    await asyncio.sleep(0)
    with pytest.raises(OverloadedError):
        async with limiter.slot():
            pass
    release.set()
    await task


async def test_limiter_releases_slot_on_exception() -> None:
    limiter = ConcurrencyLimiter(1, acquire_timeout_s=0.05)
    with pytest.raises(ValueError, match="boom"):
        async with limiter.slot():
            raise ValueError("boom")
    async with limiter.slot():  # would time out if the slot leaked
        pass


async def test_gather_bounded_keeps_order_and_collects_failures() -> None:
    async def ok(v: int) -> int:
        await asyncio.sleep(0.01 * (5 - v))  # finish in REVERSE order
        return v

    async def bad() -> int:
        raise LLMBadRequestError("nope")

    factories = [lambda v=v: ok(v) for v in range(4)] + [bad]
    results = await gather_bounded(factories, limit=2)
    assert results[:4] == [0, 1, 2, 3]
    assert isinstance(results[4], LLMBadRequestError)


async def test_run_all_or_raise_cancels_siblings() -> None:
    sibling_cancelled = asyncio.Event()

    async def slow() -> int:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise  # ALWAYS re-raise CancelledError after cleanup
        return 1

    async def failing() -> int:
        await asyncio.sleep(0.01)
        raise LLMBadRequestError("fail fast")

    started = time.perf_counter()
    with pytest.raises(LLMBadRequestError):
        await run_all_or_raise([slow, failing], limit=5)
    assert sibling_cancelled.is_set()
    assert time.perf_counter() - started < 1


async def test_timeout_cancels_the_inner_await() -> None:
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await asyncio.sleep(1)


async def test_blocking_call_offloaded_keeps_loop_responsive() -> None:
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    t = asyncio.create_task(ticker())
    await asyncio.to_thread(time.sleep, 0.1)  # time.sleep directly here would freeze the ticker
    t.cancel()
    assert ticks > 5


request_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_var", default="-")


async def test_contextvars_are_copied_into_tasks() -> None:
    request_var.set("parent")

    async def child() -> str:
        seen = request_var.get()
        request_var.set("child")  # modifies the child's COPY only
        return seen

    assert await asyncio.create_task(child()) == "parent"
    assert request_var.get() == "parent"
