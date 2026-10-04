# AI Chat Service

```text
POST /chat ──► validate (Pydantic) ──► cache / single-flight ──► redact PII
                                                        │
             ┌──────────────────────────────────────────┘
             ▼
   retry( circuit_breaker( semaphore( timeout( LLM ) ) ) )
             │
             ▼
   parse JSON ──► validate against StructuredReply ──► invalid? feed errors back to LLM (self-repair)
             │
             ▼
   ChatResponse { reply: answer | tool_call | clarification, usage, latency, cache }
```

A production-shaped FastAPI service that turns free-form LLM output into a **validated, typed
agent action**. It is small enough to read in an evening, but every piece exists because a real
AI system needs it: timeouts, retries, backpressure, streaming, request coalescing, structured
output repair, observability, tests, and a container image.

It runs **fully offline** with a deterministic fake LLM. Flip one env var to call Anthropic.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env

make check   # ruff + mypy --strict + pytest with coverage
make run     # http://localhost:8000/docs
```

With Docker:

```bash
docker compose up --build          # runtime image, non-root, healthcheck
docker build --target test .       # runs lint + types + tests inside the build
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | One request → one validated structured reply |
| POST | `/chat/batch` | Up to 20 requests concurrently; `fail_fast` = all-or-nothing, otherwise partial success |
| POST | `/chat/stream` | Server-Sent Events: `delta` chunks, then a validated `final` (or `error`) event |
| GET | `/health/live` | Liveness (process is up) |
| GET | `/health/ready` | Readiness + circuit state + LLM slots in use |

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"12 * (3 + 4)"}]}' | jq
```

```json
{
  "id": "chat_6c0b…",
  "model": "fake-llm-1",
  "reply": {
    "action": { "type": "tool_call", "tool": "calculator", "arguments": { "expression": "12 * (3 + 4)" } },
    "sentiment": "neutral",
    "language": "en"
  },
  "usage": { "input_tokens": 452, "output_tokens": 35, "total_tokens": 487 },
  "latency_ms": 0.19,
  "repair_attempts": 0,
  "cache": "miss"
}
```

Streaming: `curl -N localhost:8000/chat/stream -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"tell me something"}]}'`

Errors always look the same and carry the request id you can grep for in the logs:

```json
{ "error": { "code": "llm_timeout", "message": "LLM call exceeded 30.0s", "request_id": "71404e…", "details": {} } }
```

| Situation | HTTP | `code` |
|---|---|---|
| Invalid request body | 422 | (FastAPI default) |
| Model output never matched schema | 502 | `structured_output_invalid` |
| Provider 5xx / network error | 502 | `llm_upstream_error` |
| Provider rejected our request | 502 | `llm_bad_request` |
| Provider rate-limited us | 503 + `Retry-After` | `llm_rate_limited` |
| Too many concurrent LLM calls | 503 | `overloaded` |
| Circuit breaker open | 503 | `circuit_open` |
| LLM timeout | 504 | `llm_timeout` |
| Anything unexpected | 500 | `internal_error` (no internals leaked) |

## Project layout

```text
api/
  main.py              app factory + lifespan (startup/shutdown of shared resources)
  config.py            pydantic-settings, SecretStr, cross-field validation
  schemas.py           request/response models, discriminated union, tool registry
  errors.py            exception hierarchy (status, code, retryable)
  handlers.py          exception → JSON mapping
  middleware.py        pure-ASGI request-id + access log
  logging_config.py    JSON logs, contextvars, LogRecord factory
  dependencies.py      FastAPI DI aliases
  routes/              chat.py (chat, batch, stream), health.py
  llm/                 base.py (Protocol), fake.py, anthropic.py (raw httpx), factory.py
  services/
    chat_service.py    orchestration + self-repair loop + streaming
    resilience.py      retry/backoff/jitter, ParamSpec decorator, circuit breaker
    concurrency.py     semaphore limiter, gather vs TaskGroup, ExceptionGroup
    cache.py           TTL + LRU + single-flight with asyncio.shield
    structured.py      JSON extraction, validation, repair prompt
    redaction.py       PII redaction, asyncio.to_thread
tests/                 91 tests, ~94% branch coverage
Dockerfile             multi-stage: builder / test / runtime
.github/workflows/     CI: lint, mypy, tests (3.11 + 3.12), docker build
docs/INTERVIEW_GUIDE.md   ← read this before an interview
CONTRIBUTING.md        Git/GitHub workflow
```

## Concept map: where each topic lives

| Topic | Where to look | What to notice |
|---|---|---|
| **async/await** | `services/chat_service.py`, `llm/anthropic.py` | Every I/O call is awaited; nothing blocks the loop |
| **Timeouts** | `_complete_resilient`, `stream_events` | `asyncio.timeout` per attempt; *idle* timeout for streams |
| **Cancellation** | `resilience.py` (`finally`), `cache.py` (`shield`), tests | `CancelledError` is never swallowed; cleanup always runs |
| **Concurrency limits** | `services/concurrency.py` | Semaphore + acquire timeout = backpressure (503, not an infinite queue) |
| **gather vs TaskGroup** | `gather_bounded` vs `run_all_or_raise` | Partial success vs structured all-or-nothing |
| **ExceptionGroup / except\*** | `run_all_or_raise` | Unwrapping TaskGroup failures into one domain error |
| **Blocking work** | `services/redaction.py` | `asyncio.to_thread`, GIL trade-offs, threshold |
| **contextvars** | `logging_config.py`, `middleware.py` | Request id flows into every log line and child task |
| **Single-flight** | `services/cache.py` | 50 identical concurrent requests → 1 LLM call |
| **FastAPI DI** | `dependencies.py`, `test_unexpected_errors_…` | `Annotated[..., Depends]`, `dependency_overrides` |
| **Lifespan** | `main.py` | Connection pool opened once, closed on shutdown |
| **Streaming (SSE)** | `routes/chat.py` | `aclosing`, disconnect detection, in-band errors |
| **Pydantic v2** | `schemas.py` | `Annotated` constraints, discriminated union, `model_validator`, `computed_field`, strict-in / lenient-out |
| **Structured output** | `services/structured.py` | Extract → validate → repair loop with exact errors |
| **typing** | everywhere | `Protocol`, `TypeVar`, `ParamSpec`, `Generic`, `Literal`, `Self`, `ClassVar`, mypy `--strict` |
| **pytest** | `tests/` | Fixtures, factory fixtures, parametrize, fake clocks, `MockTransport`, `caplog`, async tests |
| **logging** | `logging_config.py` | Structured JSON, `extra=`, no PII in logs |
| **Docker** | `Dockerfile` | Layer caching, multi-stage, non-root, exec-form CMD, HEALTHCHECK |
| **Git/GitHub** | `CONTRIBUTING.md`, `.github/` | Trunk-based flow, Conventional Commits, CI gates, PR template |

## Using a real LLM

```bash
APP_LLM_PROVIDER=anthropic APP_ANTHROPIC_API_KEY=sk-ant-... make run
```

The provider sits behind the `LLMClient` Protocol, so adding OpenAI, Bedrock or a local model
is one new class plus one branch in `llm/factory.py`. Nothing else changes.

## Exercises (do these to really own the material)

1. Add a `POST /agent/run` endpoint that *executes* `tool_call` actions (safe calculator via
   `ast`, stub weather) and loops until the model returns an `answer`, with a max-steps guard.
2. Make the cache distributed (Redis) and explain what happens to single-flight across pods.
3. Add per-client rate limiting (token bucket) keyed by API key.
4. Add OpenTelemetry tracing: one span per LLM attempt with retry count and token usage.
5. Add an idempotency-key header so client retries of `/chat` don't double-charge tokens.
6. Run `locust` or `hey` against `/chat` with `delay_s=1` on the fake and watch the limiter
   return 503s. Tune `APP_LLM_MAX_CONCURRENCY` and explain the trade-off.
