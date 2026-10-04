import pytest
from pydantic import ValidationError

from api.schemas import (
    BatchItemResult,
    ChatRequest,
    Clarification,
    FinalAnswer,
    StructuredReply,
    ToolCall,
    Usage,
)
from tests.factories import reply_json


class TestChatRequest:
    def test_valid_request_strips_whitespace_and_applies_defaults(self) -> None:
        req = ChatRequest.model_validate({"messages": [{"role": "user", "content": "  hello  "}]})
        assert req.messages[0].content == "hello"
        assert req.temperature == 0.0
        assert req.cacheable is True

    @pytest.mark.parametrize(
        ("payload", "expected_fragment"),
        [
            ({"messages": []}, "at least 1 item"),
            ({"messages": [{"role": "user", "content": "   "}]}, "at least 1 character"),
            ({"messages": [{"role": "system", "content": "x"}]}, "'user' or 'assistant'"),
            ({"messages": [{"role": "user", "content": "x"}], "temperature": 1.5}, "less than or equal to 1"),
            (
                {"messages": [{"role": "user", "content": "x"}], "surprise": True},
                "Extra inputs are not permitted",
            ),
            ({"messages": [{"role": "assistant", "content": "x"}]}, "last message must have role 'user'"),
        ],
        ids=["empty", "blank-content", "bad-role", "temperature-range", "extra-field", "last-not-user"],
    )
    def test_invalid_requests(self, payload: dict[str, object], expected_fragment: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest.model_validate(payload)
        assert expected_fragment in str(exc_info.value)

    def test_cache_key_ignores_use_cache_but_not_content(self) -> None:
        a = ChatRequest.model_validate({"messages": [{"role": "user", "content": "hi"}], "use_cache": True})
        b = ChatRequest.model_validate({"messages": [{"role": "user", "content": "hi"}], "use_cache": False})
        c = ChatRequest.model_validate({"messages": [{"role": "user", "content": "hey"}]})
        assert a.cache_key() == b.cache_key() != c.cache_key()

    def test_nonzero_temperature_is_not_cacheable(self) -> None:
        req = ChatRequest.model_validate({"messages": [{"role": "user", "content": "x"}], "temperature": 0.7})
        assert req.cacheable is False

    def test_messages_are_immutable(self) -> None:
        req = ChatRequest.model_validate({"messages": [{"role": "user", "content": "x"}]})
        with pytest.raises(ValidationError):
            req.messages[0].content = "changed"  # type: ignore[misc]


class TestStructuredReply:
    @pytest.mark.parametrize(
        ("action", "expected_type"),
        [
            ({"type": "answer", "text": "hi", "confidence": 0.5}, FinalAnswer),
            ({"type": "tool_call", "tool": "calculator", "arguments": {"expression": "1+1"}}, ToolCall),
            ({"type": "clarification", "question": "which one?"}, Clarification),
        ],
    )
    def test_discriminated_union_picks_the_right_model(
        self, action: dict[str, object], expected_type: type
    ) -> None:
        reply = StructuredReply.model_validate_json(reply_json(action))
        assert isinstance(reply.action, expected_type)

    def test_unknown_discriminator_gives_precise_error(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            StructuredReply.model_validate_json(reply_json({"type": "dance"}))
        errors = exc_info.value.errors()
        assert errors[0]["type"] == "union_tag_invalid"

    def test_tool_arguments_are_validated_against_tool_schema(self) -> None:
        bad = {"type": "tool_call", "tool": "calculator", "arguments": {"expression": "__import__('os')"}}
        with pytest.raises(ValidationError, match="String should match pattern"):
            StructuredReply.model_validate_json(reply_json(bad))

    def test_tool_arguments_get_defaults(self) -> None:
        action = {"type": "tool_call", "tool": "get_weather", "arguments": {"city": "Oslo"}}
        reply = StructuredReply.model_validate_json(reply_json(action))
        assert isinstance(reply.action, ToolCall)
        assert reply.action.arguments == {"city": "Oslo", "unit": "celsius"}

    def test_extra_keys_from_llm_are_ignored(self) -> None:
        reply = StructuredReply.model_validate_json(reply_json(reasoning="chain of thought..."))
        assert "reasoning" not in reply.model_dump()

    def test_json_schema_uses_oneof_with_discriminator(self) -> None:
        schema = StructuredReply.model_json_schema()
        action_schema = schema["properties"]["action"]
        assert "discriminator" in action_schema
        assert len(action_schema["oneOf"]) == 3


def test_computed_field_is_serialized() -> None:
    assert Usage(input_tokens=3, output_tokens=4).model_dump()["total_tokens"] == 7


def test_batch_item_invariant() -> None:
    with pytest.raises(ValidationError, match="ok=True requires response"):
        BatchItemResult(index=0, ok=True)
