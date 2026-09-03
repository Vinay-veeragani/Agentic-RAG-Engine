import pytest
from pydantic import BaseModel

from agentic_rag.core.errors import ModelProviderError
from agentic_rag.generation.llm import BaseLLMProvider, extract_json


class _Answer(BaseModel):
    value: int


class _ScriptedProvider(BaseLLMProvider):
    """Returns each entry of `responses` in order, one per `complete()` call."""

    def __init__(self, responses: list[str]) -> None:
        self.model_name = "scripted"
        self._responses = list(responses)

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        return self._responses.pop(0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"value": 1}', '{"value": 1}'),
        ('```json\n{"value": 1}\n```', '{"value": 1}'),
        ('Sure, here you go:\n{"value": 1}\nHope that helps!', '{"value": 1}'),
    ],
)
def test_extract_json_handles_common_wrapping(raw: str, expected: str) -> None:
    assert extract_json(raw) == expected


@pytest.mark.asyncio
async def test_complete_structured_succeeds_on_first_try() -> None:
    provider = _ScriptedProvider(['{"value": 42}'])
    result = await provider.complete_structured(
        system_prompt="sys", user_prompt="user", schema=_Answer
    )
    assert result.value == 42


@pytest.mark.asyncio
async def test_complete_structured_recovers_after_one_bad_response() -> None:
    provider = _ScriptedProvider(["not json at all", '{"value": 7}'])
    result = await provider.complete_structured(
        system_prompt="sys", user_prompt="user", schema=_Answer
    )
    assert result.value == 7


@pytest.mark.asyncio
async def test_complete_structured_raises_after_two_bad_responses() -> None:
    provider = _ScriptedProvider(["not json", "still not json"])
    with pytest.raises(ModelProviderError):
        await provider.complete_structured(system_prompt="sys", user_prompt="user", schema=_Answer)


@pytest.mark.asyncio
async def test_complete_structured_rejects_schema_violating_json() -> None:
    provider = _ScriptedProvider(['{"value": "not an int"}', '{"value": "still not an int"}'])
    with pytest.raises(ModelProviderError):
        await provider.complete_structured(system_prompt="sys", user_prompt="user", schema=_Answer)
