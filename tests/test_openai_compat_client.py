"""vLLM / OpenAI-compatible client tests. httpx.MockTransport plays the GPU server."""

import json
import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from api.config import Settings
from api.errors import LLMBadRequestError, LLMRateLimitError, LLMServerError, LLMTimeoutError
from api.llm.base import LLMClient
from api.llm.factory import build_llm_client
from api.llm.openai_compat import OpenAICompatibleClient, sanitize_schema
from api.schemas import Message, StructuredReply
from api.services.structured import parse_structured, response_schema
from tests.factories import reply_json

MESSAGES = [Message(role="user", content="What is 2 + 2?")]
MODEL = "Qwen/Qwen3.5-4B"


def completion(content: str | None, *, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "model": MODEL,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30},
    }


def make_client(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> OpenAICompatibleClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://gpu-box:8000/v1/")
    return OpenAICompatibleClient(base_url="unused", model=MODEL, timeout_s=5, http=http, **kwargs)


def capture() -> tuple[dict[str, Any], Callable[[httpx.Request], httpx.Response]]:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion(reply_json()))

    return seen, handler


async def test_satisfies_protocol() -> None:
    assert isinstance(make_client(lambda r: httpx.Response(200)), LLMClient)


async def test_request_payload_for_qwen() -> None:
    seen, handler = capture()
    client = make_client(handler)
    await client.complete(
        system="SYS",
        messages=MESSAGES,
        temperature=0.0,
        max_tokens=512,
        response_schema=response_schema(StructuredReply),
    )
    body = seen["body"]
    assert seen["url"] == "http://gpu-box:8000/v1/chat/completions"
    assert body["model"] == MODEL
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1] == {"role": "user", "content": "What is 2 + 2?"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert (body["top_k"], body["top_p"], body["presence_penalty"]) == (20, 1.0, 0.0)
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert "discriminator" not in json.dumps(fmt)


async def test_guided_json_can_be_disabled() -> None:
    seen, handler = capture()
    await make_client(handler, guided_json=False).complete(
        system="s", messages=MESSAGES, temperature=0, max_tokens=64, response_schema={"type": "object"}
    )
    assert "response_format" not in seen["body"]


async def test_api_key_is_sent_as_bearer_token() -> None:
    seen, handler = capture()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x/v1/")
    client = OpenAICompatibleClient(base_url="unused", model=MODEL, timeout_s=5, http=http)
    await client.complete(system="s", messages=MESSAGES, temperature=0, max_tokens=64)
    assert "authorization" not in seen["headers"]

    owned = OpenAICompatibleClient(base_url="http://x/v1", model=MODEL, timeout_s=5, api_key="secret")
    assert owned._http.headers["authorization"] == "Bearer secret"
    assert str(owned._http.base_url) == "http://x/v1/"
    await owned.aclose()


async def test_parses_content_and_usage() -> None:
    client = make_client(lambda r: httpx.Response(200, json=completion(reply_json())))
    result = await client.complete(system="s", messages=MESSAGES, temperature=0, max_tokens=64)
    assert parse_structured(result.text, StructuredReply).action.type == "answer"
    assert (result.input_tokens, result.output_tokens) == (120, 30)


async def test_null_content_becomes_empty_string() -> None:
    client = make_client(lambda r: httpx.Response(200, json=completion(None)))
    result = await client.complete(system="s", messages=MESSAGES, temperature=0, max_tokens=64)
    assert result.text == ""


async def test_truncation_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    client = make_client(
        lambda r: httpx.Response(200, json=completion('{"action": {', finish_reason="length"))
    )
    with caplog.at_level(logging.WARNING):
        await client.complete(system="s", messages=MESSAGES, temperature=0, max_tokens=64)
    assert "llm_output_truncated" in caplog.messages


@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, LLMRateLimitError), (503, LLMServerError), (400, LLMBadRequestError), (404, LLMBadRequestError)],
)
async def test_status_mapping(status: int, expected: type[Exception]) -> None:
    client = make_client(lambda r: httpx.Response(status, json={"error": {"message": "x"}}))
    with pytest.raises(expected):
        await client.complete(system="s", messages=MESSAGES, temperature=0, max_tokens=64)


async def test_connection_refused_is_retryable_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMServerError, match="cannot reach vLLM") as exc_info:
        await make_client(handler).complete(system="s", messages=MESSAGES, temperature=0, max_tokens=64)
    assert exc_info.value.retryable


async def test_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow GPU", request=request)

    with pytest.raises(LLMTimeoutError):
        await make_client(handler).complete(system="s", messages=MESSAGES, temperature=0, max_tokens=64)


async def test_stream_parses_openai_sse_and_skips_reasoning() -> None:
    def frame(delta: dict[str, Any]) -> str:
        return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n\n"

    body = (
        frame({"role": "assistant"})
        + frame({"reasoning_content": "hmm, let me think"})
        + frame({"content": '{"a":'})
        + frame({"content": " 1}"})
        + "data: [DONE]\n\n"
    )
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=body.encode())

    chunks = [
        c
        async for c in make_client(handler).stream(
            system="s", messages=MESSAGES, temperature=0, max_tokens=64
        )
    ]
    assert chunks == ['{"a":', " 1}"]
    assert seen["body"]["stream"] is True


@pytest.mark.parametrize(
    ("served", "expected"),
    [([MODEL], True), (["Qwen/Qwen3.5-2B"], False)],
    ids=["match", "mismatch"],
)
async def test_verify_model(served: list[str], expected: bool) -> None:
    client = make_client(lambda r: httpx.Response(200, json={"data": [{"id": m} for m in served]}))
    assert await client.verify_model() is expected


async def test_verify_model_when_server_is_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert await make_client(handler).verify_model() is False


def test_sanitize_schema_keeps_structure_and_property_names() -> None:
    schema = {
        "title": "Root",
        "properties": {"title": {"type": "string", "title": "Title"}},
        "oneOf": [{"$ref": "#/$defs/A"}],
        "discriminator": {"propertyName": "type"},
    }
    assert sanitize_schema(schema) == {
        "properties": {"title": {"type": "string"}},  # property NAMED "title" survives
        "oneOf": [{"$ref": "#/$defs/A"}],
    }


def test_sanitized_reply_schema_still_forces_one_action() -> None:
    cleaned = sanitize_schema(StructuredReply.model_json_schema())
    assert len(cleaned["properties"]["action"]["oneOf"]) == 3
    assert {"FinalAnswer", "ToolCall", "Clarification"} <= set(cleaned["$defs"])


def test_factory_builds_vllm_client() -> None:
    settings = Settings(
        _env_file=None, llm_provider="vllm", vllm_model="Qwen/Qwen3.5-2B", vllm_enable_thinking=True
    )
    client = build_llm_client(settings)
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "Qwen/Qwen3.5-2B"
    assert client.enable_thinking is True
