"""Remote LLM provider + the provider registry.

`OpenAILLMProvider` mirrors `embeddings.providers.OpenAIEmbeddingProvider`:
a plain `httpx` call to the documented REST API rather than pulling in the
`openai` SDK for one call site. Implemented but not exercised by any test in
this environment (no API key configured) — see docs/architecture.md.
"""

from __future__ import annotations

import httpx

from agentic_rag.core.config import ProviderName, Settings
from agentic_rag.core.errors import ModelProviderError
from agentic_rag.core.retry import request_with_retry
from agentic_rag.generation.llm import BaseLLMProvider, LLMProvider

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OpenAILLMProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_MODEL) -> None:
        self.model_name = model
        self._api_key = api_key

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await request_with_retry(
                    lambda: client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json={
                            "model": self.model_name,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                    )
                )
        except httpx.TransportError as exc:
            raise ModelProviderError(
                f"OpenAI chat completion request failed after retries: {exc}"
            ) from exc
        if response.status_code != 200:
            raise ModelProviderError(
                f"OpenAI chat completion request failed: {response.status_code}",
                details={"body": response.text[:500]},
            )
        content: str = response.json()["choices"][0]["message"]["content"]
        return content


def get_llm_provider(provider: ProviderName, settings: Settings) -> LLMProvider:
    if provider == "mock":
        from agentic_rag.generation.mock import MockLLMProvider

        return MockLLMProvider()
    if provider in ("local", "ollama"):
        from agentic_rag.generation.local import OllamaLLMProvider

        return OllamaLLMProvider(settings.ollama_base_url)
    if provider == "openai":
        if not settings.openai_api_key:
            raise ModelProviderError("OPENAI_API_KEY is not configured")
        return OpenAILLMProvider(api_key=settings.openai_api_key)
    raise ModelProviderError(f"no LLM provider available for {provider!r}")
