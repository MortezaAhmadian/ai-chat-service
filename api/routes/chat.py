import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from functools import partial

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.dependencies import ChatServiceDep, SettingsDep
from api.errors import AppError
from api.handlers import to_error_body
from api.schemas import (
    BatchChatRequest,
    BatchChatResponse,
    BatchItemResult,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)
from api.services.chat_service import SSEEvent
from api.services.concurrency import gather_bounded, run_all_or_raise

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

_ERRORS: dict[int | str, dict[str, object]] = {
    502: {"model": ErrorResponse, "description": "LLM failure or invalid structured output"},
    503: {"model": ErrorResponse, "description": "Overloaded, rate limited, or circuit open"},
    504: {"model": ErrorResponse, "description": "LLM timeout"},
}


@router.post("", response_model=ChatResponse, responses=_ERRORS)
async def chat(req: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    return await service.chat(req)


@router.post("/batch", response_model=BatchChatResponse, responses=_ERRORS)
async def chat_batch(
    body: BatchChatRequest, service: ChatServiceDep, settings: SettingsDep
) -> BatchChatResponse:
    factories = [partial(service.chat, r) for r in body.requests]

    if body.fail_fast:
        responses = await run_all_or_raise(factories, limit=settings.batch_concurrency)
        return BatchChatResponse(
            results=[BatchItemResult(index=i, ok=True, response=r) for i, r in enumerate(responses)]
        )

    outcomes = await gather_bounded(factories, limit=settings.batch_concurrency)
    results: list[BatchItemResult] = []
    for i, outcome in enumerate(outcomes):
        if isinstance(outcome, BaseException):
            if not isinstance(outcome, AppError):
                logger.error("batch_item_unexpected_error", exc_info=outcome)
            results.append(BatchItemResult(index=i, ok=False, error=to_error_body(outcome)))
        else:
            results.append(BatchItemResult(index=i, ok=True, response=outcome))
    return BatchChatResponse(results=results)


@router.post(
    "/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE stream"}},
)
async def chat_stream(req: ChatRequest, request: Request, service: ChatServiceDep) -> StreamingResponse:
    async def body() -> AsyncGenerator[str, None]:
        # aclosing(): breaking out of `async for` does NOT close an async
        # generator by itself; without this, cleanup would wait for GC.
        async with aclosing(service.stream_events(req)) as events:
            try:
                async for event in events:
                    if await request.is_disconnected():
                        logger.info("client_disconnected")
                        return
                    yield event.encode()
            except AppError as exc:
                # Headers (status 200) are already sent; the only way to report
                # an error now is in-band, as an SSE event.
                yield SSEEvent("error", to_error_body(exc).model_dump_json()).encode()

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
