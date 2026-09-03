"""Local LLM provider via Ollama — no API key, runs entirely on this
machine (spec: "Ollama/local-model path where practical").

Implemented but not exercised by any test in this environment: Ollama isn't
installed/running here (see docs/architecture.md). The request/response
shape follows Ollama's documented `/api/chat` endpoint.
"""

from __future__ import annotations

import httpx

from agentic_rag.core.errors import ModelProviderError
from agentic_rag.core.retry import request_with_retry
from agentic_rag.generation.llm import BaseLLMProvider

DEFAULT_OLLAMA_MODEL = "llama3.2"


class OllamaLLMProvider(BaseLLMProvider):
    def __init__(self, base_url: str, model: str = DEFAULT_OLLAMA_MODEL) -> None:
        self.model_name = model
        self._base_url = base_url.rstrip("/")

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await request_with_retry(
                    lambda: client.post(
                        f"{self._base_url}/api/chat",
                        json={
                            "model": self.model_name,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "stream": False,
                            "format": "json",
                            "options": {"temperature": temperature, "num_predict": max_tokens},
                        },
                    )
                )
        except httpx.TransportError as exc:
            raise ModelProviderError(f"Ollama request failed after retries: {exc}") from exc
        if response.status_code != 200:
            raise ModelProviderError(
                f"Ollama request failed: {response.status_code}",
                details={"body": response.text[:500]},
            )
        return str(response.json()["message"]["content"])
