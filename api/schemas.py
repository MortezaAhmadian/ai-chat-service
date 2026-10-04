"""Pydantic v2 models: the API contract AND the LLM output contract.

Design rule (Postel's law, applied deliberately):
- Requests from OUR clients are strict  -> extra="forbid".
- Output from the LLM is parsed liberally -> extra="ignore" (models love to add
  keys), but every field we *use* is strictly validated.
"""

import hashlib
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    model_validator,
)

# ---------------------------------------------------------------------------
# Request side
# ---------------------------------------------------------------------------

Content = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["user", "assistant"]
    content: Content


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"messages": [{"role": "user", "content": "What is 12 * (3 + 4)?"}]}]
        },
    )

    messages: Annotated[list[Message], Field(min_length=1, max_length=50)]
    temperature: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    max_tokens: Annotated[int, Field(ge=16, le=4096)] = 1024
    use_cache: bool = True

    @model_validator(mode="after")
    def _last_message_from_user(self) -> Self:
        if self.messages[-1].role != "user":
            raise ValueError("the last message must have role 'user'")
        return self

    @property
    def cacheable(self) -> bool:
        # Only deterministic requests are safe to cache.
        return self.use_cache and self.temperature == 0.0

    def cache_key(self) -> str:
        raw = self.model_dump_json(exclude={"use_cache"})
        return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# LLM output contract (what the model must return)
# ---------------------------------------------------------------------------


class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: Annotated[str, StringConstraints(min_length=1, max_length=200, pattern=r"^[0-9+\-*/(). ]+$")]


class WeatherArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    unit: Literal["celsius", "fahrenheit"] = "celsius"


ToolName = Literal["calculator", "get_weather"]

# Tool registry: name -> argument schema. Adding a tool = one line here.
TOOL_ARGS: dict[str, type[BaseModel]] = {
    "calculator": CalculatorArgs,
    "get_weather": WeatherArgs,
}


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["answer"]
    text: Annotated[str, Field(min_length=1)]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["tool_call"]
    tool: ToolName
    arguments: dict[str, Any]

    @model_validator(mode="after")
    def _validate_arguments(self) -> Self:
        # Validate free-form args against the tool's own schema and normalise
        # them (defaults filled in). pydantic's ValidationError subclasses
        # ValueError, so it is reported as a normal validation error.
        args_model = TOOL_ARGS[self.tool]
        self.arguments = args_model.model_validate(self.arguments).model_dump()
        return self


class Clarification(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["clarification"]
    question: Annotated[str, Field(min_length=1)]


# Discriminated (tagged) union: pydantic reads "type" and validates against
# exactly ONE model -> O(1), precise error messages, clean JSON schema (oneOf).
AssistantAction = Annotated[FinalAnswer | ToolCall | Clarification, Field(discriminator="type")]


class StructuredReply(BaseModel):
    """The contract the LLM must satisfy. Its JSON schema is put in the prompt."""

    model_config = ConfigDict(extra="ignore")
    action: AssistantAction
    sentiment: Literal["positive", "neutral", "negative"]
    language: Annotated[str, StringConstraints(pattern=r"^[a-z]{2}$")]  # ISO 639-1


# ---------------------------------------------------------------------------
# Response side
# ---------------------------------------------------------------------------


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


CacheStatus = Literal["miss", "hit", "joined", "bypass"]


class ChatResponse(BaseModel):
    id: str
    model: str
    reply: StructuredReply
    usage: Usage
    latency_ms: float
    repair_attempts: int = 0
    cache: CacheStatus = "bypass"


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class BatchChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requests: Annotated[list[ChatRequest], Field(min_length=1, max_length=20)]
    fail_fast: bool = False


class BatchItemResult(BaseModel):
    index: int
    ok: bool
    response: ChatResponse | None = None
    error: ErrorBody | None = None

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> Self:
        if self.ok != (self.response is not None) or self.ok == (self.error is not None):
            raise ValueError("ok=True requires response only; ok=False requires error only")
        return self


class BatchChatResponse(BaseModel):
    results: list[BatchItemResult]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def succeeded(self) -> int:
        return sum(r.ok for r in self.results)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failed(self) -> int:
        return len(self.results) - self.succeeded


class ReadinessResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    llm_provider: str
    circuit: str
    llm_slots_in_use: int
