"""Provider-agnostic LLM interface.

Protocol = structural typing ("duck typing the type checker understands").
FakeLLMClient and AnthropicClient never inherit from LLMClient, yet mypy
verifies both satisfy it. This keeps the domain decoupled from vendors.
"""

from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from api.schemas import Message


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


@runtime_checkable
class LLMClient(Protocol):
    """`response_schema` is an optional JSON Schema the provider MAY enforce with
    constrained decoding (vLLM does). Providers that can't enforce it ignore it;
    we validate with Pydantic afterwards either way."""

    name: str

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResult: ...

    # NOTE: declared with plain `def` returning AsyncGenerator. An implementation
    # written as `async def ...: yield ...` matches this, because calling an
    # async generator function returns the generator immediately (no await).
    def stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]: ...

    async def aclose(self) -> None: ...
