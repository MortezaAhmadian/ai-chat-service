"""Application factory.

Run with:  uvicorn api.main:create_app --factory
A factory (instead of a module-level `app`) means importing the module has no
side effects, and tests can build isolated apps with their own settings and a
fake LLM.
"""

import logging
from collections.abc import AsyncGenerator, AsyncIterator

from fastapi import FastAPI

from api import __version__
from api.config import Settings, get_settings
from api.handlers import register_exception_handlers
from api.llm import LLMClient, build_llm_client
from api.llm.openai_compat import OpenAICompatibleClient
from api.logging_config import configure_logging
from api.middleware import RequestContextMiddleware
from api.routes import chat, health
from api.schemas import ChatResponse
from api.services.cache import SingleFlightTTLCache
from api.services.chat_service import ChatService
from api.services.concurrency import ConcurrencyLimiter
from api.services.resilience import CircuitBreaker, RetryPolicy

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, *, llm_client: LLMClient | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)

    @AsyncGenerator
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Everything that owns a resource (connection pool, semaphore) is built
        # here, inside the running event loop, and torn down on shutdown.
        llm = llm_client or build_llm_client(settings)
        app.state.chat_service = ChatService(
            llm,
            limiter=ConcurrencyLimiter(
                settings.llm_max_concurrency, acquire_timeout_s=settings.limiter_acquire_timeout_s
            ),
            breaker=CircuitBreaker(
                failure_threshold=settings.circuit_failure_threshold,
                reset_timeout_s=settings.circuit_reset_timeout_s,
            ),
            retry_policy=RetryPolicy(
                max_retries=settings.llm_max_retries,
                base_delay_s=settings.retry_base_delay_s,
                max_delay_s=settings.retry_max_delay_s,
            ),
            cache=SingleFlightTTLCache[ChatResponse](
                ttl_s=settings.cache_ttl_s, max_size=settings.cache_max_size
            ),
            llm_timeout_s=settings.llm_timeout_s,
            stream_idle_timeout_s=settings.stream_idle_timeout_s,
            max_repairs=settings.structured_max_repairs,
        )
        if isinstance(llm, OpenAICompatibleClient):
            await llm.verify_model()  # logs a clear error if the model name doesn't match vLLM
        logger.info("startup", extra={"env": settings.env, "llm_provider": llm.name})
        try:
            yield
        finally:
            await llm.aclose()  # close the HTTP connection pool gracefully
            logger.info("shutdown")

    app = FastAPI(title="AI Chat Service", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(chat.router)
    return app
