"""LLM provider interface + structured-output helper.

Every provider implements `complete()` (raw text, mirroring how every real
chat API actually works) and inherits a default `complete_structured()` from
`BaseLLMProvider` that wraps `complete()` with schema-instructed prompting,
JSON parsing, pydantic validation, and one retry with the error fed back to
the model — spec principle: "Return structured JSON/Pydantic output. Do not
return uncontrolled free-form planning text."

`MockLLMProvider` does not go through that wrapper — since it never calls a
real model, it directly introspects the requested pydantic schema and fills
it with deterministic, query-derived heuristic values (see mock.py). This
keeps every downstream agent (query analyzer, planner, expander, decomposer,
and later evidence/citation agents) working end-to-end with zero external
dependencies, the same role `MockEmbeddingProvider` and `MockReranker` play.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from agentic_rag.core.errors import ModelProviderError

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object from LLM output that may be
    wrapped in prose or a markdown code fence."""
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    match = _JSON_BLOCK.search(text)
    return match.group(0) if match else text


class LLMProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str: ...

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        temperature: float = 0.0,
    ) -> T: ...


class BaseLLMProvider:
    """Provides `complete_structured` in terms of `complete` — subclasses
    only need to implement `complete` and `model_name`."""

    model_name: str

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        raise NotImplementedError

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        temperature: float = 0.0,
    ) -> T:
        schema_instructions = (
            f"{system_prompt}\n\n"
            "Respond with ONLY a single JSON object matching this JSON Schema — "
            "no prose, no markdown fence, no explanation:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )

        raw = await self.complete(
            system_prompt=schema_instructions, user_prompt=user_prompt, temperature=temperature
        )
        try:
            return schema.model_validate(json.loads(extract_json(raw)))
        except (json.JSONDecodeError, ValidationError) as first_error:
            retry_prompt = (
                f"{user_prompt}\n\n"
                f"Your previous response was invalid: {first_error}\n"
                "Return ONLY the corrected JSON object."
            )
            raw_retry = await self.complete(
                system_prompt=schema_instructions, user_prompt=retry_prompt, temperature=temperature
            )
            try:
                return schema.model_validate(json.loads(extract_json(raw_retry)))
            except (json.JSONDecodeError, ValidationError) as second_error:
                raise ModelProviderError(
                    f"LLM did not return valid structured output after retry: {second_error}",
                    details={"raw_response": raw_retry[:1000]},
                ) from second_error
