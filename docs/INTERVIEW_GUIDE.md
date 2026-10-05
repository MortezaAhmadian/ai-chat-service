# Interview Guide: Python for AI Agent Engineering

How to use this: read a question, answer it **out loud** without looking, then compare. Every
answer points to the code in this repo so you can say *"here's how I did it and why"*, which is
what separates senior candidates from people who memorised definitions.

Level tags: 🟢 fundamentals · 🟡 mid-level · 🔴 senior / "expert probing"

---

## 1. async / await and the event loop

**🟢 Q1. What actually happens when you `await` something?**
A coroutine runs until it hits an `await` on something not yet ready. It then *suspends*, handing
control back to the event loop, which runs other ready tasks. When the awaited I/O completes, the
loop resumes the coroutine where it left off. It's cooperative multitasking: one thread, and
switches happen **only** at `await` points.

**🟢 Q2. Coroutine vs Task vs Future?**
- *Coroutine*: the object returned by calling an `async def`. Nothing runs until it's awaited or
  scheduled.
- *Task*: a coroutine wrapped and scheduled on the loop (`asyncio.create_task`). It starts running
  concurrently without being awaited.
- *Future*: a low-level placeholder for a result that will exist later. Task is a subclass.
Trap: calling `foo()` without `await` does nothing except produce a "never awaited" warning.

**🟢 Q3. When is async the right tool, and when isn't it?**
Async shines for **I/O-bound, high-concurrency** work: many simultaneous LLM/HTTP/DB calls that
mostly wait. LLM serving is the textbook case: a call takes seconds and the CPU is idle. Async
does **not** speed up CPU-bound work (tokenising huge corpora, embeddings maths in pure Python).
For that use processes, native libraries that release the GIL, or separate workers.

**🟡 Q4. You call a blocking function inside `async def`. What happens and how do you fix it?**
The whole event loop freezes: every other request stalls for that duration (a `time.sleep(2)`
blocks all users for 2 seconds). Fixes: use the async version of the library (`httpx.AsyncClient`
not `requests`), or offload with `await asyncio.to_thread(fn, ...)` /
`loop.run_in_executor(pool, fn)`. → `services/redaction.py`, `test_blocking_call_offloaded_…`.
How to detect: `PYTHONASYNCIODEBUG=1` / `loop.set_debug(True)` logs callbacks slower than 100 ms;
ruff's `ASYNC` rules flag blocking calls statically.

**🟡 Q5. `asyncio.gather` vs `asyncio.TaskGroup`?**
- `gather`: runs awaitables concurrently, returns results **in input order**. With
  `return_exceptions=True` it collects failures as values → good for *partial success* (the batch
  endpoint without `fail_fast`). Without it, the first exception propagates but **the other tasks
  keep running** (orphans).
- `TaskGroup` (3.11+): *structured concurrency*. If one child fails, all siblings are cancelled,
  the group waits for them, then raises an `ExceptionGroup`. No orphans, ever.
→ `gather_bounded` vs `run_all_or_raise` in `services/concurrency.py`.

**🔴 Q6. How does cancellation work, and what's the #1 cancellation bug?**
`task.cancel()` throws `CancelledError` into the coroutine at its current `await`. Since 3.8
`CancelledError` inherits from `BaseException`, so `except Exception` does not catch it, which is
deliberate. The #1 bug: catching it (`except BaseException`, bare `except:`) and not re-raising,
which makes the task un-cancellable and breaks timeouts and TaskGroups. Rule: you may catch it to
clean up, but **always re-raise**. Second bug: cleanup in `except` instead of `finally`. See the
circuit breaker: the half-open probe flag is reset in `finally`, and
`test_cancelled_probe_releases_the_slot` proves a cancelled probe doesn't wedge the breaker.

**🔴 Q7. `asyncio.timeout()` vs `asyncio.wait_for()`?**
Both cancel the inner work on deadline and raise `TimeoutError`. `asyncio.timeout` (3.11+) is a
context manager, so it can wrap *several* awaits and its deadline can be rescheduled.
`wait_for` wraps one awaitable. In 3.11+ `asyncio.TimeoutError is TimeoutError`.

**🔴 Q8. Total timeout vs idle timeout for streaming. Which and why?**
A long, healthy answer can stream for 60 s; a total timeout of 30 s would kill it. What you really
want to detect is *silence*. So `stream_events` applies `asyncio.timeout` **per chunk** (idle
timeout). Often you combine both: generous total, tight idle.

**🔴 Q9. What does `asyncio.shield` do and when have you used it?**
It protects an inner awaitable from cancellation of the *outer* waiter. In the single-flight cache,
many requests wait on one shared task. If the first client disconnects, its request is cancelled,
but `shield` stops that cancellation from killing the shared LLM call the other 49 clients need.
Caveat: shield doesn't make the inner task immortal; if nothing else holds a reference it can be
garbage-collected, which is why the cache stores the task in `_inflight`.

**🔴 Q10. Why must you keep a reference to tasks created with `create_task`?**
The loop holds only a weak reference. A fire-and-forget task with no strong reference can be
garbage-collected mid-execution. Store it (set/dict, or a TaskGroup) and discard on completion.

**🔴 Q11. Do you need locks in asyncio? Isn't it single-threaded?**
Code between two `await`s is atomic with respect to other tasks, so a check-then-set with no
`await` in between needs no lock (that's why `CircuitBreaker` has none; read the docstring). You
need `asyncio.Lock` when the critical section itself awaits, e.g. "check cache → await fetch →
write cache", where another task can interleave at the await. `asyncio.Lock` is **not**
thread-safe; for cross-thread use `threading.Lock` or `loop.call_soon_threadsafe`.

**🔴 Q12. Async generators: what goes wrong if you `break` out of `async for`?**
The generator is *not* closed immediately; its `finally` (e.g., closing an upstream HTTP stream)
runs only when it's garbage-collected, possibly on a different task or never. Fix:
`async with contextlib.aclosing(gen()) as g: async for ...`. → `routes/chat.py` streaming route,
and `stream_events` calls `await upstream.aclose()` in `finally`.

**🟡 Q13. What's uvloop?** A drop-in event loop built on libuv, typically 2–4× faster for network
I/O. `uvicorn[standard]` uses it automatically.

**🔴 Q14. What's `contextvars` and why not thread-locals?**
Thread-locals are shared by every task on the same thread, so request A's id would leak into
request B. `ContextVar` values are per-*context*; each Task gets a **copy** of the context at
creation. That's how the request id set in middleware appears in every log line, including inside
child tasks, without being passed around. → `logging_config.py`, `test_contextvars_are_copied_into_tasks`.

---

## 2. Concurrency models

**🟢 Q15. Threads vs processes vs asyncio?**
| | Threads | Processes | asyncio |
|---|---|---|---|
| Best for | blocking I/O with sync libs | CPU-bound | massive I/O concurrency |
| Parallel CPU? | No (GIL)* | Yes | No |
| Cost per unit | ~MBs stack | heavy, IPC/pickling | ~KBs per task |
| Switching | preemptive (OS) | preemptive (OS) | cooperative (`await`) |
*Free-threaded CPython (PEP 703, 3.13+ experimental build) changes this; mention you're aware of it.

**🟡 Q16. Explain the GIL precisely.**
The Global Interpreter Lock lets only one thread execute Python bytecode at a time per process.
It's released during blocking I/O and by many C extensions (NumPy, hashing, compression), so
threads *do* help for I/O and for native-heavy work, but not for pure-Python CPU loops.

**🟡 Q17. How do you limit concurrency, and why must you?**
`asyncio.Semaphore(n)`. Without a limit, a traffic spike fires 10,000 simultaneous LLM calls: you
blow the provider's rate limit, exhaust connection pools and memory, and every request gets slow.
→ `ConcurrencyLimiter`; `test_llm_concurrency_is_capped` proves max in-flight == limit.

**🔴 Q18. What is backpressure and how does this service implement it?**
Backpressure means pushing load back to callers instead of queueing unboundedly. A plain semaphore
is an **unbounded queue** of waiters: under sustained overload latency grows forever and requests
time out anyway, after wasting resources. The limiter puts a timeout on *acquiring* a slot; if it
can't get one, it fails fast with **503 `overloaded`** so load balancers and clients back off
(load shedding). → `test_overload_returns_503`.

**🔴 Q19. What's a cache stampede and how do you prevent it?**
When a hot key is missing or expires, N concurrent requests all miss and all call the backend.
For LLMs that's N × cost. Single-flight (a.k.a. request coalescing) makes the first request compute
and the others await the same in-flight task. → `services/cache.py`, `test_single_flight_over_http`.
Follow-up: across multiple pods you need a distributed lock or a shared cache with "lock key"
semantics; in-process single-flight only dedupes within one process.

**🔴 Q20. Why only cache when `temperature == 0`?**
With sampling, the same prompt *should* produce different outputs; caching would silently make the
system deterministic and change product behaviour. And even at temperature 0 providers aren't
perfectly deterministic. Also: the cache key must include everything that affects output (model,
system prompt version, tools, params).

---

## 3. FastAPI

**🟢 Q21. `async def` vs `def` endpoints?**
`async def` runs on the event loop: never block inside it. Plain `def` endpoints run in a
threadpool (AnyIO, default ~40 threads), so blocking code is "okay" but limited by pool size.
Common senior gotcha: putting a sync DB/HTTP call in an `async def` endpoint silently serialises
the whole server.

**🟢 Q22. How does dependency injection work in FastAPI?**
Parameters declared with `Depends(fn)` are resolved per request; dependencies can depend on others,
results are cached within a request, and `yield` dependencies give setup/teardown. Using
`Annotated[ChatService, Depends(get_chat_service)]` aliases (`ChatServiceDep`) keeps signatures
clean and type-safe. Tests replace them via `app.dependency_overrides`.
→ `dependencies.py`, `test_unexpected_errors_do_not_leak_internals`.

**🟡 Q23. What's the lifespan and why not create the HTTP client per request?**
`lifespan` runs startup code before serving and teardown after. Creating an `httpx.AsyncClient`
per request throws away connection pooling and keep-alive: every call pays TCP + TLS handshakes
(~100+ ms) and you can exhaust sockets. Create once, share, close on shutdown. → `main.py`.
(`@app.on_event` is deprecated.)

**🟡 Q24. How does validation produce 422s? How do you customise errors?**
FastAPI builds a Pydantic model from the signature; failures raise `RequestValidationError` → 422
with locations. Domain errors here are a hierarchy (`AppError`) mapped centrally by one handler, so
every error has the same shape and a `request_id`. → `errors.py`, `handlers.py`.

**🔴 Q25. Pure ASGI middleware vs `BaseHTTPMiddleware`?**
`BaseHTTPMiddleware` is convenient but wraps responses through an extra stream and task layer:
overhead on every request, awkward with streaming responses, and a history of contextvar
propagation problems. Pure ASGI middleware is a callable `(scope, receive, send)` that wraps
`send` to observe the status and inject headers. → `middleware.py`.

**🔴 Q26. Why does the error middleware validate `x-request-id`?**
Anything from the client that ends up in logs is an injection vector (newlines forging log lines,
huge values). Allow-list the charset and length; otherwise generate one.

**🔴 Q27. Streaming: what can't you do once the first byte is sent?**
Change the status code or headers. Errors after that must be reported *in-band* (an SSE `error`
event). You also can't transparently retry. → `test_stream_reports_invalid_output_in_band`.

**🟡 Q28. SSE vs WebSockets for LLM streaming?**
SSE: one-directional server→client over plain HTTP, auto-reconnect in browsers, works through most
proxies, perfect for token streaming. WebSockets: bidirectional, needed for interrupting
generation mid-stream, voice, or multi-turn real-time agents. Set `X-Accel-Buffering: no` so Nginx
doesn't buffer the stream.

**🔴 Q29. How do you handle client disconnects during a long LLM call?**
Detect via `request.is_disconnected()` (or rely on Starlette cancelling the response task), make
sure cancellation propagates to the upstream call so you stop paying for tokens nobody will read,
and close upstream streams in `finally`. Exception: shared single-flight work is shielded because
other clients still need it.

**🟡 Q30. Liveness vs readiness?**
Liveness: "is the process alive?" Failing it restarts the container, so it must *never* depend on
external services. Readiness: "should I get traffic?" Senior nuance (see `/health/ready`): don't
fail readiness because a *shared* dependency (the LLM) is down; every pod would go unready at once
and you'd turn a degraded service into a total outage. Report `degraded` instead.

**🔴 Q31. How do you run it in production? Workers?**
`uvicorn --factory` behind a load balancer; in Kubernetes typically **one process per container**
and scale with replicas (cleaner metrics, memory limits, and graceful shutdown). On a VM, gunicorn
with uvicorn workers or `uvicorn --workers N`. Remember in-process state (cache, breaker,
semaphore) is **per worker**; your global concurrency is `limit × workers × replicas`.

---

## 4. Pydantic v2

**🟢 Q32. What changed in v2?** Core rewritten in Rust (pydantic-core), 5–50× faster.
`parse_obj` → `model_validate`, `.dict()` → `model_dump()`, `.json()` → `model_dump_json()`,
`@validator` → `@field_validator`/`@model_validator`, `Config` class → `model_config = ConfigDict(...)`.

**🟡 Q33. `model_validate_json` vs `model_validate(json.loads(...))`?**
`model_validate_json` parses and validates in one Rust pass: faster, less memory, and malformed
JSON becomes a normal `ValidationError` (`json_invalid`) instead of a separate `JSONDecodeError`
path. → `services/structured.py`.

**🟡 Q34. Field vs model validators; `mode="before"` vs `"after"`?**
Field validators handle one field; model validators see the whole object (cross-field rules like
"last message must be from the user"). `before` runs on raw input (coercion/normalisation);
`after` runs on validated, typed data (invariants). → `ChatRequest._last_message_from_user`,
`ToolCall._validate_arguments`.

**🔴 Q35. What's a discriminated union and why is it ideal for agent outputs?**
`Annotated[A | B | C, Field(discriminator="type")]`: Pydantic reads the tag and validates against
exactly one model. Versus a plain union (tries each member, "smart mode"): it's faster,
unambiguous, gives precise errors (`union_tag_invalid`), and emits a JSON Schema `oneOf` +
`discriminator` that LLMs follow well. An agent step is naturally a tagged union:
answer | tool_call | clarification. → `schemas.py`, `test_unknown_discriminator_gives_precise_error`.

**🔴 Q36. Why `extra="forbid"` on requests but `extra="ignore"` on LLM output?**
Clients are programmers: reject typos loudly. LLMs routinely add harmless keys ("reasoning");
rejecting those burns a repair round-trip for nothing. Be strict on fields you *use*, lenient on
noise. Being able to justify this trade-off is a strong signal.

**🔴 Q37. How do you validate tool arguments whose schema depends on the tool name?**
Tool registry (`TOOL_ARGS`) plus a model validator that validates `arguments` against the tool's
own model and normalises defaults. Alternative: nested discriminated union per tool. Note that
Pydantic's `ValidationError` subclasses `ValueError`, so raising it inside a validator is reported
cleanly.

**🟡 Q38. Strict vs lax mode?** Lax (default) coerces `"5"` → `5`; strict mode (`ConfigDict(strict=True)`
or `Field(strict=True)`) refuses. Use strict where coercion hides bugs (IDs, money).

**🟡 Q39. `computed_field`, `frozen`, `model_copy`?** `computed_field` adds derived values to
serialisation (`total_tokens`). `frozen=True` makes instances immutable and hashable
(`Message`). `model_copy(update=...)` creates a modified copy without re-validation (used to set
`cache` status). Caution: `update` skips validation.

**🟡 Q40. How do you manage configuration?** `pydantic-settings`: typed env vars with a prefix,
`.env` support, `SecretStr` so keys never appear in logs or reprs, and validators that **fail at
startup** if config is inconsistent (provider=anthropic without key). → `config.py`.

---

## 5. Structured output and LLM-specific engineering

**🟡 Q41. How do you reliably get structured output from an LLM?**
Layers, strongest first: (1) provider-native structured outputs / tool-use with a JSON schema
(constrained decoding), (2) schema in the system prompt, (3) robust extraction (strip fences/prose),
(4) **validate with Pydantic**, (5) on failure, a **repair loop** that feeds the exact validation
errors back, (6) bounded attempts, then a clear 502. Even with constrained decoding you still
validate: schema-valid ≠ semantically valid. → `chat_service._generate`.

**🔴 Q42. Why send the validation errors back instead of just retrying?**
A blind retry repeats the same mistake at the same cost. Showing the model *its own output* plus
*precise* errors ("`sentiment`: Input should be 'positive', 'neutral' or 'negative'") converts most
failures in one round. Track `repair_attempts` as a metric: a rising rate signals prompt or model
regression.

**🔴 Q43. Which errors are retryable?**
Retry: timeouts, connection errors, 5xx, 429 (respecting `Retry-After`), provider "overloaded"
(e.g., 529). Don't retry: 400/401/403/404/422; retrying a bad request just burns money. Encoded
once as `retryable` on the exception class. → `errors.py`.

**🔴 Q44. Exponential backoff: why jitter?**
Without jitter, all clients that failed together retry together, recreating the spike (thundering
herd). Full jitter, `uniform(0, min(cap, base·2^attempt))`, spreads them out.
→ `RetryPolicy.compute_delay`, `test_full_jitter_stays_within_bounds`.

**🔴 Q45. Explain the circuit breaker and its interaction with retries.**
Closed → counts upstream failures → Open after a threshold (fail fast, don't touch upstream) →
after a cool-down, Half-open: allow **one** probe; success closes, failure re-opens. Order matters:
`retry(breaker(call))`, so once open, `CircuitOpenError` is non-retryable and retries stop
immediately. Only *upstream health* errors count, not our own bad requests or rate limits.

**🔴 Q46. Your upstream returns 429. What status do you return to your client and why?**
503 with `Retry-After`. Your client didn't exceed *their* limit; *your* shared quota is exhausted,
which is a temporary server-side condition. 429 would wrongly tell them they're misbehaving.

**🔴 Q47. How do you make LLM endpoints idempotent?**
Accept an `Idempotency-Key` header, store the response (or in-flight marker) keyed by it with a TTL,
and return the stored result on repeat. This prevents duplicate charges and duplicate side effects
(tool executions!) when clients retry after a network blip.

**🔴 Q48. What do you log for LLM calls, and what must you never log?**
Log: request id, model, latency, token usage, retries, repair attempts, action type, error codes,
and cost. Don't log raw prompts/outputs by default (PII, secrets, compliance); if you need them for
evals, use a separate, access-controlled, retention-limited store. This service redacts PII
*before* sending to the provider. → `redaction.py`.

**🔴 Q49. Prompt injection: how does this design limit the blast radius?**
The model can only produce one of three typed actions; tools are an allow-list (`Literal`), and
arguments are validated (the calculator pattern rejects `__import__('os')`). Never `eval` model
output; execute tools with least privilege; treat retrieved/tool content as data, never
instructions; require confirmation for destructive actions.

**🔴 Q50. How would you evaluate this service?**
Golden dataset of inputs → expected action type/tool/args; metrics: schema-valid rate, repair
rate, action accuracy, latency p50/p95/p99, cost per request. Run on every prompt/model change in
CI (with recorded or real calls), plus online monitoring and sampled human review.

**🟡 Q51. Tokens and cost.** Count input + output tokens (output usually priced higher); cap
`max_tokens`; trim conversation history; cache aggressively; use smaller models for simple routing.

---

## 6. typing

**🟢 Q52. Why type hints in Python at all?** Catch bugs before runtime (mypy/pyright), better
IDE help, executable documentation, and libraries (FastAPI, Pydantic) *use* them at runtime to
validate and generate OpenAPI.

**🟡 Q53. `Protocol` vs `ABC`?**
ABC = nominal: implementations must inherit. Protocol = structural: anything with matching methods
qualifies, checked statically (and at runtime with `@runtime_checkable`, which only checks
method *names*). `LLMClient` is a Protocol, so providers don't depend on our base class.
→ `llm/base.py`, `test_satisfies_protocol`.

**🔴 Q54. How do you type an async generator in a Protocol?**
Declare `def stream(...) -> AsyncGenerator[str, None]` (plain `def`, no `async`). An implementation
`async def stream(...): yield ...` returns the generator immediately when called, so it matches.
Declaring `async def` in the protocol would mean "coroutine returning a generator", which is wrong.

**🔴 Q55. What's `ParamSpec` for?**
Typing decorators that preserve the wrapped function's exact signature:
`Callable[P, Awaitable[T]] -> Callable[P, Awaitable[T]]` with `*args: P.args, **kwargs: P.kwargs`.
→ `with_retry` in `resilience.py`.

**🟡 Q56. `TypeVar`, bound, `Generic`?**
`M = TypeVar("M", bound=BaseModel)` makes `parse_structured(text, StructuredReply)` return
`StructuredReply`, not `BaseModel`. `SingleFlightTTLCache(Generic[T])` is a generic class
instantiated as `SingleFlightTTLCache[ChatResponse]`. (3.12 offers `def f[T](...)` syntax.)

**🟡 Q57. `Literal`, `Annotated`, `Self`, `ClassVar`, `Final`?**
`Literal["a","b"]`: exact values (also become JSON-schema enums). `Annotated[T, meta]`: attach
metadata (constraints, `Depends`) without changing the type. `Self`: methods returning the same
class (validators). `ClassVar`: class-level attributes not per-instance fields (error `status_code`).

**🔴 Q58. Variance, briefly?**
`list[Dog]` is not a `list[Animal]` (lists are mutable → invariant); `Sequence[Dog]` *is* a
`Sequence[Animal]` (read-only → covariant). That's why function parameters should accept
`Sequence`/`Mapping` rather than `list`/`dict` (see `messages: Sequence[Message]`).

**🟡 Q59. What does `mypy --strict` buy you?** Disallows untyped defs, implicit `Any`, missing
return types, untyped decorators, etc. This repo passes it, which is worth saying in an interview.

---

## 7. pytest

**🟢 Q60. Fixtures: scopes and why?**
`function` (default, isolation), `class`, `module`, `session` (expensive shared resources). Wider
scope = faster but risk of state leaking. Async fixtures need a matching event-loop scope.

**🟡 Q61. How do you test FastAPI with async code?**
`httpx.AsyncClient(transport=httpx.ASGITransport(app=app))`, no server needed. Gotcha:
ASGITransport **does not run lifespan**; run `app.router.lifespan_context(app)` in the fixture.
→ `tests/conftest.py`.

**🟡 Q62. Mock vs fake vs stub?**
Mock: records calls and asserts interactions (brittle if overused). Stub: returns canned answers.
Fake: a working lightweight implementation (`FakeLLMClient`, scriptable and concurrency-aware).
Prefer fakes at boundaries you own and `httpx.MockTransport` for HTTP: you test real request and
response semantics without the network. → `tests/test_anthropic_client.py`.

**🔴 Q63. How do you test time-dependent code (TTL, backoff, breaker) without sleeping?**
Inject the clock and sleep functions (`clock=time.monotonic`, `sleep=asyncio.sleep` as default
params) and pass fakes in tests. Tests stay fast and deterministic. → `FakeClock`, `fake_sleep`.

**🔴 Q64. How do you prove a concurrency limit actually works?**
Measure it: the fake LLM records `max_in_flight`; fire many concurrent requests and assert it
equals the limit. Same for single-flight (assert exactly one LLM call). Asserting only "200 OK"
proves nothing about concurrency.

**🟡 Q65. `parametrize`, `ids`, factory fixtures, `caplog`, `monkeypatch`?**
All used in `tests/`. A factory fixture (`make_client`) builds apps with custom settings inside a
single test while an `AsyncExitStack` guarantees teardown.

**🔴 Q66. A test is flaky. What does that tell you?** Order-dependent or timing-dependent
tests indicate shared state or real sleeps. Fix the design (inject clocks, isolate state), don't
add retries to tests.

**🟡 Q67. Testing pyramid for an LLM service?** Many unit tests (parsing, retry, cache), a solid
layer of integration tests with a fake LLM (full HTTP stack), a few contract tests against the
real provider (nightly, real key), plus offline evals on a golden set.

---

## 8. Logging and observability

**🟢 Q68. Why structured (JSON) logs?** Machines query them: `level=ERROR AND code=llm_timeout`
group-by model. Free-text logs require regex archaeology.

**🟡 Q69. `logger.exception` vs `logger.error`?** `exception` (inside `except`) attaches the
traceback. Always pass `extra={...}` for fields, not f-strings: lazy formatting and searchable keys.

**🔴 Q70. How does one request id appear in every log line without passing it around?**
`ContextVar` set in middleware + a custom `LogRecord` factory that stamps `request_id` onto every
record, so even third-party loggers and pytest's `caplog` see it. Returned to clients in
`x-request-id`; propagate it to upstream calls for distributed tracing.

**🔴 Q71. Logs vs metrics vs traces?**
Logs: discrete events with detail. Metrics: cheap aggregated numbers (p95 latency, error rate,
tokens/min, breaker state) for dashboards and alerts. Traces: one request's path across services
with timing per span (each LLM attempt a span). OpenTelemetry unifies all three.

---

## 9. Docker

**🟢 Q72. Image vs container? Layers?** Image: immutable layered filesystem + metadata. Container:
a running instance with a writable layer. Each Dockerfile instruction creates a cached layer;
a change invalidates that layer and everything after it.

**🟡 Q73. Why copy `requirements.txt` before the source code?** Layer caching: dependencies
reinstall only when requirements change, not on every code edit. Builds go from minutes to seconds.

**🟡 Q74. Multi-stage builds?** Build tools and caches stay in the builder stage; the runtime image
copies only the virtualenv and code: smaller, fewer CVEs, faster pulls. This repo also has a `test`
stage so CI can run the whole suite inside the same environment.

**🔴 Q75. Exec form vs shell form `CMD`, and why it matters for graceful shutdown.**
Shell form (`CMD uvicorn ...`) runs `/bin/sh -c`, so sh is PID 1 and may not forward SIGTERM;
`docker stop` waits 10 s then SIGKILLs, and in-flight LLM requests die mid-response. Exec form
makes uvicorn PID 1, it receives SIGTERM, stops accepting, finishes in-flight requests, runs
lifespan teardown. `init: true` / `tini` adds zombie reaping.

**🟡 Q76. Why non-root?** Container escape or RCE as root is far worse. `USER app` costs nothing.

**🟡 Q77. Secrets in Docker?** Never `ENV API_KEY=...` or `COPY .env` into the image (layers are
forever and `docker history` shows them). Inject at runtime (env from orchestrator secrets,
mounted files); use BuildKit `--secret` for build-time secrets. `.dockerignore` excludes `.env`.

**🟡 Q78. HEALTHCHECK, `PYTHONUNBUFFERED`, `PYTHONDONTWRITEBYTECODE`?** Health status for
orchestrators; unbuffered stdout so logs appear immediately; no `.pyc` clutter in the image.

**🔴 Q79. `slim` vs `alpine` for Python?** Alpine uses musl libc; many wheels aren't available, so
packages compile from source (slow, large, sometimes subtly broken). `-slim` (Debian, glibc) is the
usual choice; distroless for maximum minimalism.

---

## 10. Git / GitHub

**🟢 Q80. Merge vs rebase?** Merge preserves history with a merge commit; rebase replays your commits
on top of the target for a linear history. Golden rule: don't rebase commits others have based work
on; if you must update your own pushed branch, use `--force-with-lease`, never plain `--force`.

**🟡 Q81. Squash merging, pros and cons?** One clean commit per PR on `main`, easy reverts; but
loses granular history and makes stacked branches painful.

**🟡 Q82. Conventional Commits?** `feat:`, `fix:`, `refactor:`, `!` for breaking: readable history,
automated changelogs and semver. → `CONTRIBUTING.md`.

**🔴 Q83. A bug appeared somewhere in the last 200 commits. Go.** `git bisect` with
`git bisect run pytest -q tests/test_x.py`: binary search, about 8 steps.

**🔴 Q84. You pushed an API key.** Rotate/revoke it **immediately** (it's compromised the moment it's
public), then purge it from history (`git filter-repo`), then add prevention (pre-commit secret
scanning, GitHub push protection).

**🟡 Q85. What should CI enforce before merge?** Lint, format, strict types, tests with coverage
threshold, image build, ideally security scans; branch protection requires them plus review.
→ `.github/workflows/ci.yml`.

**🟡 Q86. `revert` vs `reset`?** `revert` adds an inverse commit (safe on shared branches);
`reset` moves the branch pointer (rewrites history; local-only). `reflog` recovers almost anything.

---

## 11. Python fundamentals interviewers still ask

**🟡 Q87. Mutable default arguments?** `def f(x=[])` shares one list across calls. Use `None`
sentinel; in Pydantic use `Field(default_factory=dict)` (Pydantic copies defaults, but say it).

**🟡 Q88. `@dataclass(frozen=True, slots=True)` vs Pydantic model?** Dataclass: fast internal value
object, no validation (`LLMResult`). Pydantic: validation/serialisation at trust boundaries (API,
LLM output, config). Validate at the edges, use plain types inside.

**🟡 Q89. Context managers / async context managers?** `__enter__/__exit__` or
`__aenter__/__aexit__`; `@asynccontextmanager` for generator-style. Guarantee cleanup.
→ `ConcurrencyLimiter.slot()`, lifespan.

**🔴 Q90. Exception chaining: `raise X from exc` vs `from None`?** `from exc` keeps the cause in the
traceback (debuggable), used everywhere when translating `httpx` errors into domain errors.
`from None` hides it deliberately.

**🔴 Q91. `ExceptionGroup` and `except*`?** Raised by TaskGroup when multiple children fail.
`except* AppError as eg` handles only the matching subgroup; others propagate. → `run_all_or_raise`.

**🟡 Q92. `functools.cache`/`lru_cache` pitfalls?** Arguments must be hashable; unbounded `cache`
can leak memory; on methods it holds `self` alive. Don't use on async functions: it caches the
*coroutine object*, which can only be awaited once.

---

## 11b. Self-hosted models (vLLM)

**🟡 Q93. Why serve the model with vLLM instead of loading it with `transformers` inside FastAPI?**
GPU inference is blocking compute: inside an async server it freezes the event loop. A naive
`model.generate()` also handles one request at a time. vLLM gives continuous batching and
PagedAttention (efficient KV-cache memory), so many concurrent requests share the GPU, plus an
OpenAI-compatible API. The API process stays CPU-only and stateless; model server and API
scale and restart independently. → `api/llm/openai_compat.py`.

**🔴 Q94. What is constrained (guided) decoding and why does it matter for small models?**
At each step the server masks out tokens that would violate a grammar compiled from your JSON
Schema, so the output is always parseable and schema-shaped. A 4B model is much less reliable at
free-form JSON than a frontier model; constraining turns "usually valid" into "always
structurally valid". You still validate with Pydantic: constraints don't check semantics, and
some keywords (`discriminator`, some regex features) aren't supported by grammar backends,
which is why the schema is sanitised first.

**🔴 Q95. Hybrid "thinking" models in a structured-output pipeline?**
Decide explicitly per request (`enable_thinking`) rather than trusting the model default, which
varies across sizes. Thinking improves hard reasoning but adds latency, tokens and the risk of
truncation before the JSON. Strip `<think>` blocks before parsing and never feed reasoning back
into history.

**🟡 Q96. How do you size `max-model-len` and concurrency on one GPU?**
VRAM = weights + KV cache. The KV cache needed grows with context length × concurrent sequences.
Lower `--max-model-len` to what you actually use, skip unused encoders, then load-test: increase
API-side concurrency until throughput plateaus and p95 latency rises. That knee is your limit.

## 12. System-design scenarios (whiteboard)

**🔴 S1. "Traffic goes 10× tomorrow."** Horizontal scale (stateless pods), move cache to Redis,
global rate limiting/token budget per tenant, queue for non-interactive (batch) work, provider
fallback (secondary model on breaker open), autoscaling on in-flight requests not CPU (CPU is idle
while waiting on LLMs), and load-shedding so p99 stays bounded.

**🔴 S2. "Provider is down for 20 minutes."** Breaker opens → fail fast in milliseconds, readiness
reports degraded (pods stay in rotation), alert fires, optional fallback model or cached answers,
clear 503 with `Retry-After` to clients. Recovery via half-open probe, no manual restart.

**🔴 S3. "Responses are sometimes truncated JSON."** Probably hitting `max_tokens`: check
`stop_reason`, raise `max_tokens` or shrink the schema; the repair loop masks it at extra cost, so
track repair rate as a metric to detect it.

**🔴 S4. "Turn this into an agent that executes tools."** Loop: model → validated action → if
tool_call, execute in a sandbox with timeout → append the tool result as a message → repeat, with
max steps, total token/cost budget, loop detection, idempotent tools, human approval for side
effects, and a trace of every step.

**🔴 S5. "Latency p99 is 40 s while p50 is 3 s."** Look at queueing (limiter wait time metric),
retry storms (retries count per request), long outputs (tokens), slow tail of the provider;
mitigations: streaming for perceived latency, hedged requests, smaller model for routing,
tighter timeouts with fallback.

---

## 13. Live-coding drills (do them without looking)

1. Write `retry_async` with exponential backoff + full jitter that never swallows `CancelledError`.
2. Write a bounded `gather` that preserves input order.
3. Implement single-flight for an async function.
4. Write a Pydantic discriminated union for `answer | tool_call` and parse a JSON string with it.
5. Write a FastAPI endpoint that streams SSE and closes its upstream generator on disconnect.
6. Write a pytest test proving a semaphore limits concurrency to 3.
7. Write a pure-ASGI middleware that adds `x-request-id`.
8. Write a multi-stage Dockerfile for a FastAPI app running as non-root.

## 14. Red flags vs senior signals

| Red flag | Senior signal |
|---|---|
| `requests` inside `async def` | Knows what blocks the loop and how to detect it |
| `except Exception: pass` around awaits | Talks about cancellation, `finally`, re-raising |
| "Retry everything 3 times" | Classifies errors, backoff + jitter, breaker, idempotency |
| `json.loads` + `dict["key"]` on LLM output | Schema, validation, repair loop, metrics on failure rate |
| New HTTP client per request | Lifespan-managed pooled client |
| Unbounded `gather` of 10k calls | Semaphore + load shedding + backpressure |
| Logs the full prompt with user data | Redaction, structured logs, correlation ids |
| Tests that `sleep(2)` | Injected clocks, fakes, deterministic concurrency tests |
| `CMD python app.py` as root | Exec form, non-root, healthcheck, graceful shutdown |
