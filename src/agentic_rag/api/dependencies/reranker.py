from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from agentic_rag.core.config import Settings, get_settings
from agentic_rag.retrieval.reranking import Reranker, get_reranker

_reranker_instances: dict[str, Reranker] = {}


def get_default_reranker(settings: Annotated[Settings, Depends(get_settings)]) -> Reranker:
    key = settings.reranker_provider
    if key not in _reranker_instances:
        _reranker_instances[key] = get_reranker(settings.reranker_provider)
    return _reranker_instances[key]


RerankerDep = Annotated[Reranker, Depends(get_default_reranker)]
