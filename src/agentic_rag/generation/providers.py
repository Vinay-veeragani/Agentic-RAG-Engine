"""Remote LLM provider + the provider registry.

`OpenAICompatibleLLMProvider` mirrors
`embeddings.providers.OpenAIEmbeddingProvider`: a plain `httpx` call to the
documented REST API rather than pulling in a vendor SDK for one call site.
Parameterized by base URL + model rather than one class per vendor, since
OpenAI, Groq, and a growing list of others all expose the same
`/chat/completions` request/response shape (Groq's docs describe their API
as "OpenAI SDK compatible" for exactly this reason) — `OpenAILLMProvider`
and `GroqLLMProvider` are just this with different defaults.
"""

from __future__ import annotations

import httpx

from agentic_rag.core.config import ProviderName, Settings
from agentic_rag.core.errors import ModelProviderError
from agentic_rag.core.retry import request_with_retry
from agentic_rag.generation.llm import BaseLLMProvider, LLMProvider

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class OpenAICompatibleLLMProvider(BaseLLMProvider):
    def __init__(self, *, api_key: str, base_url: str, model: str, provider_label: str) -> None:
        self.model_name = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._provider_label = provider_label

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
                        f"{self._base_url}/chat/completions",
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
                f"{self._provider_label} chat completion request failed after retries: {exc}"
            ) from exc
        if response.status_code != 200:
            raise ModelProviderError(
                f"{self._provider_label} chat completion request failed: {response.status_code}",
                details={"body": response.text[:500]},
            )
        content: str = response.json()["choices"][0]["message"]["content"]
        return content


class OpenAILLMProvider(OpenAICompatibleLLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_MODEL) -> None:
        super().__init__(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            model=model,
            provider_label="OpenAI",
        )


class GroqLLMProvider(OpenAICompatibleLLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_GROQ_MODEL) -> None:
        super().__init__(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model=model,
            provider_label="Groq",
        )


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
    if provider == "groq":
        if not settings.groq_api_key:
            raise ModelProviderError("GROQ_API_KEY is not configured")
        return GroqLLMProvider(api_key=settings.groq_api_key)
    raise ModelProviderError(f"no LLM provider available for {provider!r}")
