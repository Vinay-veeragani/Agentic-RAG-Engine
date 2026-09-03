"""Local embedding provider using sentence-transformers (CPU, no API key).

Default model is `all-MiniLM-L6-v2` — 384 dimensions, matching
`storage.models.EMBEDDING_DIMENSIONS`. Model weights download from
Hugging Face Hub on first use and are cached locally after that; no
network access is required on subsequent runs.
"""

from __future__ import annotations

import asyncio
from typing import Any

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIMENSIONS = 384


class LocalEmbeddingProvider:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: object | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return DEFAULT_DIMENSIONS

    def _get_model(self) -> Any:  # sentence-transformers ships no py.typed stubs
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]
