from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from agentic_rag.core.config import Settings, get_settings
from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.embeddings.cache import CachedEmbeddingProvider
from agentic_rag.embeddings.providers import get_embedding_provider
from agentic_rag.storage.cache import get_cache

_provider_instances: dict[str, EmbeddingProvider] = {}


def get_default_embedding_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingProvider:
    key = settings.embedding_provider
    if key not in _provider_instances:
        base = get_embedding_provider(settings.embedding_provider, settings)
        cache = get_cache(settings.redis_url)
        _provider_instances[key] = CachedEmbeddingProvider(base, cache)
    return _provider_instances[key]


EmbeddingProviderDep = Annotated[EmbeddingProvider, Depends(get_default_embedding_provider)]
