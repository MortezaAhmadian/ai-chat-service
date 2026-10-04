"""Provider client tests with httpx.MockTransport: no network, real HTTP semantics."""

import json
from collections.abc import Callable

import httpx
import pytest

from api.errors import LLMBadRequestError, LLMRateLimitError, LLMServerError, LLMTimeoutError
from api.llm.anthropic import AnthropicClient
from api.llm.base import LLMClient
from api.schemas import Message

MESSAGES = [Message(role="user", content="hi")]


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> AnthropicClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.test")
    return AnthropicClient(
        api_key="sk-test", model="claude-test", base_url="https://api.test", timeout_s=5, http=http
    )


async def test_satisfies_protocol() -> None:
    assert isinstance(make_client(lambda r: httpx.Response(200)), LLMClient)


async def test_complete_sends_correct_request_and_parses_response() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "claude-test",
                "content": [{"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}],
                "usage": {"input_tokens": 7, "output_tokens": 2},
            },
        )

    result = await make_client(handler).complete(
        system="sys", messages=MESSAGES, temperature=0, max_tokens=64
    )
    assert result.text == "Hello"
    assert (result.input_tokens, result.output_tokens) == (7, 2)
    headers = seen["headers"]
    assert isinstance(headers, dict)
    assert headers["x-api-key"] == "sk-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert seen["body"] == {
        "model": "claude-test",
        "system": "sys",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0,
        "max_tokens": 64,
    }


@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, LLMRateLimitError), (500, LLMServerError), (529, LLMServerError), (400, LLMBadRequestError)],
)
async def test_status_mapping(status: int, expected: type[Exception]) -> None:
    client = make_client(lambda r: httpx.Response(status, headers={"retry-after": "4"}, json={}))
    with pytest.raises(expected) as exc_info:
        await client.complete(system="s", messages=MESSAGES, temperature=0, max_tokens=16)
    if isinstance(exc_info.value, LLMRateLimitError):
        assert exc_info.value.retry_after == 4.0


async def test_network_timeout_maps_to_domain_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMTimeoutError):
        await make_client(handler).complete(system="s", messages=MESSAGES, temperature=0, max_tokens=16)


async def test_stream_parses_sse() -> None:
    def frame(event: dict[str, object]) -> str:
        return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    body = "".join(
        [
            frame({"type": "message_start", "message": {}}),
            frame(
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}
            ),
            frame({"type": "ping"}),
            frame({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}}),
            frame({"type": "message_stop"}),
        ]
    )
    client = make_client(lambda r: httpx.Response(200, content=body.encode()))
    chunks = [c async for c in client.stream(system="s", messages=MESSAGES, temperature=0, max_tokens=16)]
    assert chunks == ["Hel", "lo"]


async def test_owned_http_client_is_closed() -> None:
    client = AnthropicClient(api_key="k", model="m", base_url="https://api.test", timeout_s=1)
    await client.aclose()
    assert client._http.is_closed
