"""End-to-end HTTP tests: real app, real middleware, fake LLM."""

import asyncio
import json
import logging

import httpx
import pytest
from fastapi import FastAPI

from api.dependencies import get_chat_service
from api.errors import LLMRateLimitError, LLMServerError
from api.llm.fake import FakeLLMClient
from tests.conftest import ClientFactory
from tests.factories import chat_body, parse_sse, reply_json


class TestChatHappyPath:
    async def test_answer(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/chat", json=chat_body())
        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"]["action"]["type"] == "answer"
        assert data["usage"]["total_tokens"] == data["usage"]["input_tokens"] + data["usage"]["output_tokens"]
        assert data["id"].startswith("chat_")

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("12 * (3 + 4)", ("tool_call", "calculator")),
            ("What is the weather in Lisbon?", ("tool_call", "get_weather")),
            ("help", ("clarification", None)),
        ],
    )
    async def test_agent_actions(
        self, client: httpx.AsyncClient, content: str, expected: tuple[str, str | None]
    ) -> None:
        action = (await client.post("/chat", json=chat_body(content))).json()["reply"]["action"]
        assert action["type"] == expected[0]
        assert action.get("tool") == expected[1]

    async def test_request_id_is_generated_and_propagated(
        self, client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        generated = await client.post("/chat", json=chat_body())
        assert len(generated.headers["x-request-id"]) == 32

        resp = await client.post(
            "/chat", json=chat_body(use_cache=False), headers={"x-request-id": "trace-abc_123"}
        )
        assert resp.headers["x-request-id"] == "trace-abc_123"
        records = [r for r in caplog.records if getattr(r, "request_id", None) == "trace-abc_123"]
        assert {r.getMessage() for r in records} >= {"chat_completed", "request_completed"}

    async def test_malicious_request_id_is_replaced(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/chat", json=chat_body(), headers={"x-request-id": "bad id\ninjected"})
        assert resp.headers["x-request-id"] != "bad id\ninjected"

    async def test_pii_is_redacted_before_reaching_llm(
        self, client: httpx.AsyncClient, fake_llm: FakeLLMClient
    ) -> None:
        await client.post("/chat", json=chat_body("email me at jane.doe@example.com please"))
        assert "[EMAIL]" in fake_llm.calls[-1][-1].content
        assert "jane.doe" not in fake_llm.calls[-1][-1].content


class TestValidation:
    async def test_422_on_bad_body(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/chat", json={"messages": []})
        assert resp.status_code == 422
        assert resp.json()["detail"][0]["loc"] == ["body", "messages"]

    async def test_openapi_documents_error_models(self, client: httpx.AsyncClient) -> None:
        spec = (await client.get("/openapi.json")).json()
        assert {"502", "503", "504"} <= set(spec["paths"]["/chat"]["post"]["responses"])


class TestStructuredOutputRepair:
    async def test_repairs_invalid_output(self, client: httpx.AsyncClient, fake_llm: FakeLLMClient) -> None:
        fake_llm.enqueue("I think the answer is Paris.", reply_json())
        resp = await client.post("/chat", json=chat_body())
        assert resp.status_code == 200
        assert resp.json()["repair_attempts"] == 1
        second_call = fake_llm.calls[1]
        assert second_call[-2].role == "assistant"
        assert "did not match the required JSON schema" in second_call[-1].content

    async def test_gives_up_after_max_repairs(
        self, client: httpx.AsyncClient, fake_llm: FakeLLMClient
    ) -> None:
        fake_llm.enqueue(*(["nope"] * 3))
        resp = await client.post("/chat", json=chat_body())
        assert resp.status_code == 502
        err = resp.json()["error"]
        assert err["code"] == "structured_output_invalid"
        assert err["details"]["attempts"] == 3
        assert err["request_id"] == resp.headers["x-request-id"]


class TestResilienceOverHttp:
    async def test_transient_upstream_errors_are_retried(
        self, client: httpx.AsyncClient, fake_llm: FakeLLMClient
    ) -> None:
        fake_llm.enqueue(LLMServerError("500"), LLMServerError("503"), reply_json())
        resp = await client.post("/chat", json=chat_body())
        assert resp.status_code == 200
        assert len(fake_llm.calls) == 3

    async def test_upstream_rate_limit_maps_to_503_with_retry_after(
        self, client: httpx.AsyncClient, fake_llm: FakeLLMClient
    ) -> None:
        fake_llm.enqueue(*[LLMRateLimitError("429", retry_after=2.0)] * 3)
        resp = await client.post("/chat", json=chat_body())
        assert resp.status_code == 503
        assert resp.headers["retry-after"] == "2"
        assert resp.json()["error"]["code"] == "llm_rate_limited"

    async def test_timeout_maps_to_504(self, make_client: ClientFactory) -> None:
        client, _ = await make_client(FakeLLMClient(delay_s=0.5), llm_timeout_s=0.02, llm_max_retries=0)
        resp = await client.post("/chat", json=chat_body())
        assert resp.status_code == 504
        assert resp.json()["error"]["code"] == "llm_timeout"

    async def test_circuit_opens_and_readiness_degrades(self, make_client: ClientFactory) -> None:
        client, llm = await make_client(circuit_failure_threshold=2, llm_max_retries=0)
        for _ in range(2):
            assert (await client.post("/chat", json=chat_body("trigger __upstream__ now"))).status_code == 502
        calls_before = len(llm.calls)
        resp = await client.post("/chat", json=chat_body())
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "circuit_open"
        assert len(llm.calls) == calls_before  # failed fast, LLM untouched

        ready = await client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "degraded"


class TestConcurrencyOverHttp:
    async def test_llm_concurrency_is_capped(self, make_client: ClientFactory) -> None:
        client, llm = await make_client(FakeLLMClient(delay_s=0.03), llm_max_concurrency=2)
        bodies = [chat_body(f"distinct question number {i}", use_cache=False) for i in range(8)]
        responses = await asyncio.gather(*(client.post("/chat", json=b) for b in bodies))
        assert all(r.status_code == 200 for r in responses)
        assert llm.max_in_flight == 2

    async def test_overload_returns_503(self, make_client: ClientFactory) -> None:
        client, _ = await make_client(
            FakeLLMClient(delay_s=0.2), llm_max_concurrency=1, limiter_acquire_timeout_s=0.02
        )
        r1, r2 = await asyncio.gather(
            client.post("/chat", json=chat_body("first question here", use_cache=False)),
            client.post("/chat", json=chat_body("second question here", use_cache=False)),
        )
        assert sorted([r1.status_code, r2.status_code]) == [200, 503]

    async def test_cache_hit(self, client: httpx.AsyncClient, fake_llm: FakeLLMClient) -> None:
        first = await client.post("/chat", json=chat_body())
        second = await client.post("/chat", json=chat_body())
        assert (first.json()["cache"], second.json()["cache"]) == ("miss", "hit")
        assert len(fake_llm.calls) == 1

    async def test_single_flight_over_http(self, make_client: ClientFactory) -> None:
        client, llm = await make_client(FakeLLMClient(delay_s=0.05))
        responses = await asyncio.gather(*(client.post("/chat", json=chat_body()) for _ in range(5)))
        assert len(llm.calls) == 1
        assert sorted(r.json()["cache"] for r in responses) == ["joined"] * 4 + ["miss"]


class TestBatch:
    async def test_partial_failure(self, client: httpx.AsyncClient) -> None:
        body = {
            "requests": [
                chat_body("first normal question"),
                chat_body("please __error__ this one"),
                chat_body("third normal question"),
            ]
        }
        resp = await client.post("/chat/batch", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert (data["succeeded"], data["failed"]) == (2, 1)
        assert [r["ok"] for r in data["results"]] == [True, False, True]
        assert data["results"][1]["error"]["code"] == "llm_bad_request"

    async def test_fail_fast(self, client: httpx.AsyncClient) -> None:
        body = {"requests": [chat_body("fine question here"), chat_body("x __error__ y")], "fail_fast": True}
        resp = await client.post("/chat/batch", json=body)
        assert resp.status_code == 502
        assert resp.json()["error"]["code"] == "llm_bad_request"

    async def test_batch_size_limit(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/chat/batch", json={"requests": [chat_body()] * 21})
        assert resp.status_code == 422


class TestStreaming:
    async def test_stream_emits_deltas_then_validated_final(self, client: httpx.AsyncClient) -> None:
        async with client.stream("POST", "/chat/stream", json=chat_body()) as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            text = (await resp.aread()).decode()
        events = parse_sse(text)
        deltas = "".join(json.loads(d)["text"] for e, d in events if e == "delta")
        assert events[-1][0] == "final"
        assert json.loads(deltas) == json.loads(events[-1][1])

    async def test_stream_reports_invalid_output_in_band(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/chat/stream", json=chat_body("give me __invalid__ output"))
        assert resp.status_code == 200  # headers were already sent
        last_event, data = parse_sse(resp.text)[-1]
        assert last_event == "error"
        assert json.loads(data)["code"] == "structured_output_invalid"


class TestHealthAndLifecycle:
    async def test_liveness_and_readiness(self, client: httpx.AsyncClient) -> None:
        assert (await client.get("/health/live")).json() == {"status": "ok"}
        ready = (await client.get("/health/ready")).json()
        assert ready["status"] == "ok"
        assert ready["circuit"] == "closed"

    async def test_lifespan_closes_llm_client(self, settings, fake_llm: FakeLLMClient) -> None:  # type: ignore[no-untyped-def]
        from api.main import create_app

        app = create_app(settings, llm_client=fake_llm)
        async with app.router.lifespan_context(app):
            assert fake_llm.closed is False
        assert fake_llm.closed is True

    async def test_unexpected_errors_do_not_leak_internals(self, app: FastAPI) -> None:
        class ExplodingService:
            async def chat(self, req: object) -> None:
                raise RuntimeError("db password is hunter2")

        app.dependency_overrides[get_chat_service] = lambda: ExplodingService()
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/chat", json=chat_body())
        app.dependency_overrides.clear()
        assert resp.status_code == 500
        assert resp.json()["error"]["code"] == "internal_error"
        assert "hunter2" not in resp.text


async def test_lifespan_verifies_vllm_model(caplog: pytest.LogCaptureFixture) -> None:
    from api.llm.openai_compat import OpenAICompatibleClient
    from api.main import create_app
    from tests.conftest import make_settings

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"id": "other-model"}]})),
        base_url="http://gpu/v1/",
    )
    llm = OpenAICompatibleClient(base_url="unused", model="Qwen/Qwen3.5-4B", timeout_s=1, http=http)
    app = create_app(make_settings(), llm_client=llm)
    with caplog.at_level(logging.ERROR):
        async with app.router.lifespan_context(app):
            pass
    assert "vllm_model_mismatch" in caplog.messages
    await http.aclose()
