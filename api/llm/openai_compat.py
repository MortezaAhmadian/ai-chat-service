"""Client for any OpenAI-compatible /v1/chat/completions server.

Built for vLLM serving Qwen3.5 on a local GPU, but works with SGLang,
`transformers serve`, llama.cpp server, LM Studio, etc.

Why talk to vLLM over HTTP instead of loading the model in this process?
- GPU inference is blocking, compute-heavy work. Inside an async web server it
  would freeze the event loop for every request.
- vLLM does continuous batching: many concurrent requests share the GPU
  efficiently. Our async service just fires concurrent HTTP calls.
- The API process stays small and restarts in a second; the model server
  stays warm with weights loaded. Each scales independently.

Qwen3.5-specific behaviour handled here:
- Thinking mode: we send chat_template_kwargs.enable_thinking explicitly
  (default False) instead of relying on the model's default, which differs
  between Qwen3.5 sizes. Reasoning text would break JSON parsing and burn tokens.
- Constrained decoding: with `response_format=json_schema`, vLLM masks invalid
  tokens during sampling so the output is always schema-shaped JSON. For a
  small (2B/4B) model this is the difference between "usually valid" and
  "always valid". We still validate with Pydantic afterwards.
"""

import json
import logging
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import httpx

from api.errors import LLMServerError, LLMTimeoutError
from api.llm.base import LLMResult
from api.llm.http_errors import raise_for_status
from api.schemas import Message

logger = logging.getLogger(__name__)

# JSON-Schema keywords that are OpenAPI extensions or metadata. Grammar
# backends (xgrammar / guidance / outlines) either ignore or reject them.
_UNSUPPORTED_SCHEMA_KEYS = frozenset({"discriminator", "title", "examples"})


def sanitize_schema(node: Any) -> Any:
    """Recursively drop keys constrained-decoding backends don't understand.

    Pydantic emits `discriminator` for tagged unions; it's useful in docs but
    not part of JSON Schema proper. The `oneOf` + `const` "type" fields still
    force the model to pick exactly one action shape.
    Only dict KEYS in _UNSUPPORTED_SCHEMA_KEYS are removed when they are schema
    keywords, never property names inside "properties".
    """
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                cleaned[key] = {name: sanitize_schema(sub) for name, sub in value.items()}
            elif key in _UNSUPPORTED_SCHEMA_KEYS:
                continue
            else:
                cleaned[key] = sanitize_schema(value)
        return cleaned
    if isinstance(node, list):
        return [sanitize_schema(item) for item in node]
    return node


class OpenAICompatibleClient:
    name = "vllm"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float,
        api_key: str | None = None,
        guided_json: bool = True,
        enable_thinking: bool = False,
        top_p: float = 1.0,
        top_k: int = 20,
        presence_penalty: float = 0.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.guided_json = guided_json
        self.enable_thinking = enable_thinking
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self._owns_http = http is None
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        # base_url must end with "/" so relative paths ("chat/completions")
        # are appended to "/v1/" instead of replacing it.
        self._http = http or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            # Generous pool: vLLM batches concurrent requests on the GPU, so
            # parallel connections are exactly what we want.
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
        )

    # ------------------------------------------------------------------ helpers

    def _payload(
        self,
        system: str,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
        response_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            # OpenAI format: the system prompt is just the first message.
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "max_tokens": max_tokens,
            # vLLM extensions (non-OpenAI fields are accepted at the top level):
            "top_k": self.top_k,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if response_schema is not None and self.guided_json:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_reply",
                    "schema": sanitize_schema(response_schema),
                    "strict": True,
                },
            }
        return payload

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        raise_for_status(resp, provider="vllm")

    # --------------------------------------------------------------- interface

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        payload = self._payload(system, messages, temperature, max_tokens, response_schema)
        try:
            resp = await self._http.post("chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("vLLM request timed out") from exc
        except httpx.TransportError as exc:
            # Connection refused usually means vLLM is not running / wrong URL.
            raise LLMServerError(
                f"cannot reach vLLM: {exc.__class__.__name__}",
                details={"base_url": str(self._http.base_url)},
            ) from exc

        self._raise_for_status(resp)
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        if choice.get("finish_reason") == "length":
            # Truncated output is the #1 cause of "invalid JSON" with local
            # models. The repair loop will retry, but the log tells you the
            # real fix: raise max_tokens or keep thinking disabled.
            logger.warning("llm_output_truncated", extra={"max_tokens": max_tokens})
        usage = data.get("usage") or {}
        return LLMResult(
            text=text,
            model=data.get("model", self.model),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
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
        payload = self._payload(system, messages, temperature, max_tokens, response_schema)
        payload["stream"] = True
        try:
            async with self._http.stream("POST", "chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    self._raise_for_status(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise LLMServerError("vLLM stream error", details=chunk)
                    for choice in chunk.get("choices", []):
                        # `reasoning_content` (if a reasoning parser is enabled)
                        # is deliberately skipped: only the answer is streamed.
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            yield content
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("vLLM stream timed out") from exc
        except httpx.TransportError as exc:
            raise LLMServerError(f"cannot reach vLLM: {exc.__class__.__name__}") from exc

    async def verify_model(self) -> bool:
        """Startup check: is vLLM reachable and serving the model we expect?

        Logs instead of crashing: the model server may still be loading weights
        when this API starts (common with docker compose), and the circuit
        breaker + retries handle that case at request time.
        """
        try:
            resp = await self._http.get("models", timeout=5.0)
            resp.raise_for_status()
            served = [m.get("id") for m in resp.json().get("data", [])]
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("vllm_unreachable_at_startup", extra={"error": exc.__class__.__name__})
            return False
        if self.model not in served:
            logger.error(
                "vllm_model_mismatch",
                extra={"configured": self.model, "served": served},
            )
            return False
        logger.info("vllm_model_verified", extra={"model": self.model})
        return True

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()
