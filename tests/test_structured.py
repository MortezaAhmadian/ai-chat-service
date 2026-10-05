import pytest
from pydantic import ValidationError

from api.schemas import StructuredReply
from api.services.structured import (
    build_repair_prompt,
    build_system_prompt,
    extract_json_block,
    parse_structured,
)
from tests.factories import reply_json


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Sure! Here is the JSON:\n{"a": 1}\nHope that helps.',
    ],
    ids=["plain", "json-fence", "bare-fence", "prose-wrapped"],
)
def test_extract_json_block(raw: str) -> None:
    assert extract_json_block(raw) == '{"a": 1}'


def test_parse_structured_happy_path() -> None:
    reply = parse_structured(f"```json\n{reply_json()}\n```", StructuredReply)
    assert reply.action.type == "answer"


def test_invalid_json_is_a_validation_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        parse_structured("not json at all", StructuredReply)
    assert exc_info.value.errors()[0]["type"] == "json_invalid"


def test_repair_prompt_contains_field_locations() -> None:
    with pytest.raises(ValidationError) as exc_info:
        parse_structured(reply_json(sentiment="ecstatic"), StructuredReply)
    prompt = build_repair_prompt(exc_info.value)
    assert "sentiment" in prompt
    assert "ONLY the corrected JSON" in prompt


def test_system_prompt_embeds_schema_and_is_cached() -> None:
    first = build_system_prompt(StructuredReply)
    assert '"discriminator"' in first
    assert build_system_prompt(StructuredReply) is first  # functools.cache


@pytest.mark.parametrize(
    "raw",
    [
        '<think>\nThe user wants {braces} explained.\n</think>\n\n{"a": 1}',
        'Okay, let me reason about {x}.\n</think>\n{"a": 1}',  # template opened <think> in the prompt
        '<think></think>{"a": 1}',
    ],
    ids=["full-block", "close-tag-only", "empty-think"],
)
def test_reasoning_is_stripped_before_json_extraction(raw: str) -> None:
    assert extract_json_block(raw) == '{"a": 1}'
