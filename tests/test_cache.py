import asyncio

import pytest

from api.services.cache import SingleFlightTTLCache


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def cache(clock: Clock) -> SingleFlightTTLCache[str]:
    return SingleFlightTTLCache[str](ttl_s=10, max_size=2, clock=clock)


class Counter:
    def __init__(self, delay: float = 0.0, fail: bool = False) -> None:
        self.calls = 0
        self.delay = delay
        self.fail = fail

    async def __call__(self) -> str:
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("upstream broke")
        return f"value-{self.calls}"


async def test_miss_then_hit(cache: SingleFlightTTLCache[str]) -> None:
    fn = Counter()
    assert await cache.get_or_compute("k", fn) == ("value-1", "miss")
    assert await cache.get_or_compute("k", fn) == ("value-1", "hit")
    assert fn.calls == 1


async def test_entries_expire(cache: SingleFlightTTLCache[str], clock: Clock) -> None:
    fn = Counter()
    await cache.get_or_compute("k", fn)
    clock.now = 10.0
    assert await cache.get_or_compute("k", fn) == ("value-2", "miss")


async def test_lru_eviction(cache: SingleFlightTTLCache[str]) -> None:
    fn = Counter()
    await cache.get_or_compute("a", fn)
    await cache.get_or_compute("b", fn)
    await cache.get_or_compute("a", fn)  # touch a -> b is now least recently used
    await cache.get_or_compute("c", fn)
    assert len(cache) == 2
    _, outcome = await cache.get_or_compute("b", fn)
    assert outcome == "miss"


async def test_single_flight_coalesces_concurrent_misses(cache: SingleFlightTTLCache[str]) -> None:
    fn = Counter(delay=0.02)
    results = await asyncio.gather(*(cache.get_or_compute("k", fn) for _ in range(10)))
    assert fn.calls == 1
    assert {v for v, _ in results} == {"value-1"}
    assert sorted(o for _, o in results) == ["joined"] * 9 + ["miss"]


async def test_failures_are_shared_but_not_cached(cache: SingleFlightTTLCache[str]) -> None:
    fn = Counter(delay=0.01, fail=True)
    results = await asyncio.gather(*(cache.get_or_compute("k", fn) for _ in range(3)), return_exceptions=True)
    assert fn.calls == 1
    assert all(isinstance(r, RuntimeError) for r in results)
    fn.fail = False
    assert (await cache.get_or_compute("k", fn))[1] == "miss"


async def test_cancelled_waiter_does_not_cancel_shared_work(cache: SingleFlightTTLCache[str]) -> None:
    fn = Counter(delay=0.05)
    leader = asyncio.create_task(cache.get_or_compute("k", fn))
    await asyncio.sleep(0.01)
    follower = asyncio.create_task(cache.get_or_compute("k", fn))
    await asyncio.sleep(0)
    leader.cancel()  # e.g. first client disconnected
    with pytest.raises(asyncio.CancelledError):
        await leader
    assert await follower == ("value-1", "joined")
    assert fn.calls == 1
