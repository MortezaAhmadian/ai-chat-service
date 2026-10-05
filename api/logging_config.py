"""Structured JSON logging with request-id correlation.

Concepts:
- contextvars: per-request (per-task) state that is safe under asyncio.
  Each asyncio Task gets a *copy* of the context at creation time, so a
  request id set in middleware is visible in every coroutine and child task.
- LogRecord factory: injects request_id into EVERY record, so any handler
  (including pytest's caplog) sees it, not just ours.
- `extra=` fields become top-level JSON keys -> queryable in Loki/Datadog/ELK.
"""

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}
_factory_installed = False
_HANDLER_NAME = "ai-chat-service"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key != "request_id":
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _install_record_factory() -> None:
    global _factory_installed
    if _factory_installed:  # idempotent: create_app() may run many times in tests
        return
    previous = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        record.request_id = request_id_var.get()
        return record

    logging.setLogRecordFactory(factory)
    _factory_installed = True


def configure_logging(level: str = "INFO", *, json_logs: bool = True) -> None:
    _install_record_factory()
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s")
        )
    handler.set_name(_HANDLER_NAME)
    root = logging.getLogger()
    # Replace only OUR handler (idempotent across create_app() calls). Clearing
    # every root handler would also remove handlers that other code installed,
    # e.g. pytest's caplog or an APM agent.
    for existing in list(root.handlers):
        if existing.get_name() == _HANDLER_NAME:
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
