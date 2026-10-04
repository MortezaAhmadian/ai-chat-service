"""Async TTL + LRU cache with single-flight request coalescing.

Scenario: 50 identical requests arrive at once while the cache is cold.
Naive cache -> 50 LLM calls (cache stampede). Single-flight -> 1 LLM call,
49 callers await the same Task.

Cancellation semantics: the computation runs in its OWN task and callers
await it through asyncio.shield(). If one client disconnects, its request is
cancelled, but the shared computation keeps running for everyone else and
still populates the cache.
"""

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Generic, Literal, TypeVar

T = TypeVar("T")
Outcome = Literal["hit", "joined", "miss"]


class SingleFlightTTLCache(Generic[T]):
    def __init__(
        self,
        *,
        ttl_s: float,
        max_size: int = 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_s = ttl_s
        self.max_size = max_size
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._inflight: dict[str, asyncio.Task[T]] = {}

    def __len__(self) -> int:
        return len(self._data)

    def _get_fresh(self, key: str) -> tuple[bool, T | None]:
        entry = self._data.get(key)
        if entry is None:
            return False, None
        expires_at, value = entry
        if expires_at <= self._clock():
            del self._data[key]
            return False, None
        self._data.move_to_end(key)  # LRU bookkeeping
        return True, value

    def _store(self, key: str, value: T) -> None:
        self._data[key] = (self._clock() + self.ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)  # evict least-recently used

    async def get_or_compute(self, key: str, fn: Callable[[], Awaitable[T]]) -> tuple[T, Outcome]:
        found, value = self._get_fresh(key)
        if found:
            return value, "hit"  # type: ignore[return-value]

        task = self._inflight.get(key)
        outcome: Outcome = "joined"
        if task is None:
            outcome = "miss"
            task = asyncio.create_task(self._compute(key, fn), name=f"cache:{key[:12]}")
            task.add_done_callback(_mark_exception_retrieved)
            self._inflight[key] = task
        return await asyncio.shield(task), outcome

    async def _compute(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        try:
            value = await fn()
            self._store(key, value)  # only successes are cached
            return value
        finally:
            self._inflight.pop(key, None)


def _mark_exception_retrieved(task: asyncio.Task[object]) -> None:
    # If every waiter was cancelled, nobody awaits the task; touching
    # .exception() prevents the "Task exception was never retrieved" warning.
    if not task.cancelled():
        task.exception()
