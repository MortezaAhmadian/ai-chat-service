"""Central exception -> HTTP response mapping. Never leak internals to clients."""

import logging
import math

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.errors import AppError, LLMRateLimitError
from api.logging_config import request_id_var
from api.schemas import ErrorBody, ErrorResponse

logger = logging.getLogger(__name__)


def _request_id(request: Request | None = None) -> str:
    if request is not None:
        rid = getattr(request.state, "request_id", None)
        if isinstance(rid, str):
            return rid
    return request_id_var.get()


def to_error_body(exc: BaseException, request: Request | None = None) -> ErrorBody:
    if isinstance(exc, AppError):
        return ErrorBody(
            code=exc.code, message=exc.message, request_id=_request_id(request), details=exc.details
        )
    return ErrorBody(code="internal_error", message="internal server error", request_id=_request_id(request))


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
        logger.log(level, "app_error", extra={"code": exc.code, "status": exc.status_code})
        headers: dict[str, str] = {}
        if isinstance(exc, LLMRateLimitError) and exc.retry_after is not None:
            headers["Retry-After"] = str(math.ceil(exc.retry_after))
        body = ErrorResponse(error=to_error_body(exc, request))
        return JSONResponse(status_code=exc.status_code, content=body.model_dump(), headers=headers)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error")  # full traceback goes to logs only
        body = ErrorResponse(error=to_error_body(exc, request))
        return JSONResponse(status_code=500, content=body.model_dump())
