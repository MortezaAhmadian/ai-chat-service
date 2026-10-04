"""Shared fixtures.

Fixture graph: settings -> fake_llm -> app (with lifespan running) -> client.
Override `settings` in a test module/class (or use the `make_client` factory)
to test a different configuration.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from api.config import Settings
from api.llm.fake import FakeLLMClient
from api.main import create_app

BASE_TEST_SETTINGS: dict[str, Any] = {
    "_env_file": None,  # never read a developer's local .env in tests
    "env": "test",
    "log_json": False,
    "llm_timeout_s": 2.0,
    "llm_max_retries": 2,
    "retry_base_delay_s": 0.0,
    "retry_max_delay_s": 0.01,
}


def make_settings(**overrides: Any) -> Settings:
    return Settings(**{**BASE_TEST_SETTINGS, **overrides})


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
async def app(settings: Settings, fake_llm: FakeLLMClient) -> AsyncIterator[FastAPI]:
    application = create_app(settings, llm_client=fake_llm)
    # httpx.ASGITransport does NOT run lifespan events; run them explicitly.
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


ClientFactory = Callable[..., Awaitable[tuple[httpx.AsyncClient, FakeLLMClient]]]


@pytest.fixture
async def make_client() -> AsyncIterator[ClientFactory]:
    """Factory fixture: build clients with custom settings inside one test."""
    async with AsyncExitStack() as stack:

        async def _make(
            llm: FakeLLMClient | None = None, *, raise_app_exceptions: bool = True, **overrides: Any
        ) -> tuple[httpx.AsyncClient, FakeLLMClient]:
            llm = llm or FakeLLMClient()
            application = create_app(make_settings(**overrides), llm_client=llm)
            await stack.enter_async_context(application.router.lifespan_context(application))
            transport = httpx.ASGITransport(app=application, raise_app_exceptions=raise_app_exceptions)
            c = await stack.enter_async_context(
                httpx.AsyncClient(transport=transport, base_url="http://test")
            )
            return c, llm

        yield _make
