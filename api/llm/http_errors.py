"""HTTP status -> domain error mapping shared by every HTTP-based provider."""

import httpx

from api.errors import LLMBadRequestError, LLMRateLimitError, LLMServerError

RETRYABLE_STATUS = frozenset({500, 502, 503, 504, 529})


def parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # could be an HTTP-date; keep it simple


def raise_for_status(resp: httpx.Response, *, provider: str) -> None:
    status = resp.status_code
    if status < 400:
        return
    details = {"provider": provider, "status": status, "body": resp.text[:500]}
    if status == 429:
        raise LLMRateLimitError(
            f"{provider} rate limit",
            retry_after=parse_retry_after(resp.headers.get("retry-after")),
            details=details,
        )
    if status in RETRYABLE_STATUS:
        raise LLMServerError(f"{provider} returned {status}", details=details)
    raise LLMBadRequestError(f"{provider} rejected request ({status})", details=details)
