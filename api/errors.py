"""Domain exception hierarchy.

Each error knows its HTTP status, a stable machine-readable `code`, and whether
it is safe to retry. Routes never build error responses by hand; one handler
maps AppError -> JSON (see handlers.py).
"""

from typing import Any, ClassVar


class AppError(Exception):
    status_code: ClassVar[int] = 500
    code: ClassVar[str] = "internal_error"
    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}


class OverloadedError(AppError):
    """We could not get a concurrency slot in time (backpressure)."""

    status_code = 503
    code = "overloaded"


class CircuitOpenError(AppError):
    """Upstream is considered down; fail fast instead of piling on."""

    status_code = 503
    code = "circuit_open"


class LLMError(AppError):
    status_code = 502
    code = "llm_error"


class LLMTimeoutError(LLMError):
    status_code = 504
    code = "llm_timeout"
    retryable = True


class LLMServerError(LLMError):
    """5xx / transport failures from the provider."""

    status_code = 502
    code = "llm_upstream_error"
    retryable = True


class LLMRateLimitError(LLMError):
    """Provider returned 429.

    We answer our client with 503 (not 429): the *client* did nothing wrong,
    *our* upstream quota is exhausted.
    """

    status_code = 503
    code = "llm_rate_limited"
    retryable = True

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.retry_after = retry_after


class LLMBadRequestError(LLMError):
    """Provider rejected our request (4xx other than 429). Retrying won't help."""

    status_code = 502
    code = "llm_bad_request"


class StructuredOutputError(LLMError):
    """The model kept returning output that does not match our schema."""

    status_code = 502
    code = "structured_output_invalid"
