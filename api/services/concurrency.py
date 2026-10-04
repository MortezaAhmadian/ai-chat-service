"""Bounded concurrency primitives.

- ConcurrencyLimiter: caps in-flight LLM calls (protects provider quota and
  our memory). Waiting for a slot has its own timeout -> backpressure: under
  overload we shed load with 503 instead of queueing forever.
- gather_bounded: run many jobs, collect results AND failures (partial success).
- run_all_or_raise: structured concurrency with TaskGroup; first failure
  cancels all siblings (all-or-nothing).
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from typing import TypeVar

from api.errors import AppError, OverloadedError

T = TypeVar("T")
E = TypeVar("E", bound=BaseException)


class ConcurrencyLimiter:
    def __init__(self, limit: int, *, acquire_timeout_s: float) -> None:
        # Since Python 3.10 asyncio primitives are not bound to a loop at
        # construction time, but we still create this inside the app lifespan.
        self._sem = asyncio.Semaphore(limit)
        self.limit = limit
        self.acquire_timeout_s = acquire_timeout_s
        self.in_use = 0

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        try:
            async with asyncio.timeout(self.acquire_timeout_s):
                await self._sem.acquire()
        except TimeoutError as exc:
            raise OverloadedError("too many concurrent LLM requests", details={"limit": self.limit}) from exc
        self.in_use += 1
        try:
            yield
        finally:
            self.in_use -= 1
            self._sem.release()


async def gather_bounded(
    factories: Iterable[Callable[[], Awaitable[T]]], *, limit: int
) -> list[T | BaseException]:
    """Results come back in INPUT order (gather guarantees it), failures included."""
    sem = asyncio.Semaphore(limit)

    async def run(factory: Callable[[], Awaitable[T]]) -> T:
        async with sem:
            return await factory()

    return await asyncio.gather(*(run(f) for f in factories), return_exceptions=True)


def first_leaf(group: BaseExceptionGroup[E]) -> E:
    exc: BaseException = group
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return exc  # type: ignore[return-value]


async def run_all_or_raise(factories: Iterable[Callable[[], Awaitable[T]]], *, limit: int) -> list[T]:
    """All-or-nothing. On the first AppError, TaskGroup cancels the remaining
    tasks, waits for them to finish cancelling, then raises an ExceptionGroup.
    We unwrap it so the API reports one clean domain error."""
    sem = asyncio.Semaphore(limit)

    async def run(factory: Callable[[], Awaitable[T]]) -> T:
        async with sem:
            return await factory()

    error: AppError | None = None
    try:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(run(f)) for f in factories]
    except* AppError as group:
        error = first_leaf(group)
    if error is not None:
        raise error
    return [t.result() for t in tasks]
