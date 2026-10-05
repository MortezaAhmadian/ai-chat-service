"""Anthropic Messages API client built directly on httpx.AsyncClient.

Why raw httpx instead of the SDK? To show (and be able to explain):
- connection pooling + keep-alive (one shared AsyncClient for the app lifetime)
- explicit timeouts (connect vs read)
- mapping HTTP status codes to retryable / non-retryable domain errors
- parsing Server-Sent Events for streaming
In production you may well use the official SDK; the concepts are identical.
"""

import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import httpx

from api.errors import LLMServerError, LLMTimeoutError
from api.llm.base import LLMResult
from api.llm.http_errors import raise_for_status
from api.schemas import Message

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClient:
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_s: float,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _payload(
        self, system: str, messages: Sequence[Message], temperature: float, max_tokens: int
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        raise_for_status(resp, provider="anthropic")

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,  # schema is in the prompt; not enforced here
    ) -> LLMResult:
        payload = self._payload(system, messages, temperature, max_tokens)
        try:
            resp = await self._http.post("/v1/messages", json=payload, headers=self._headers)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("provider request timed out") from exc
        except httpx.TransportError as exc:
            raise LLMServerError(f"transport error: {exc.__class__.__name__}") from exc

        self._raise_for_status(resp)
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return LLMResult(
            text=text,
            model=data.get("model", self._model),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        payload = self._payload(system, messages, temperature, max_tokens) | {"stream": True}
        try:
            async with self._http.stream("POST", "/v1/messages", json=payload, headers=self._headers) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    self._raise_for_status(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:].strip())
                    etype = event.get("type")
                    if etype == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")
                    elif etype == "error":
                        raise LLMServerError("provider stream error", details=event)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("provider stream timed out") from exc
        except httpx.TransportError as exc:
            raise LLMServerError(f"transport error: {exc.__class__.__name__}") from exc

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()
