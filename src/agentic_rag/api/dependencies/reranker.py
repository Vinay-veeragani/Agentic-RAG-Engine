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


async def warm_up_default_reranker(settings: Settings) -> None:
    """Called once at app startup (see api/main.py's lifespan) so a real
    model-backed reranker loads before the first request needs it, not
    during it. Populates the same cache `get_default_reranker` reads from,
    so the request-time instance is the already-warmed one."""
    key = settings.reranker_provider
    reranker = _reranker_instances.setdefault(key, get_reranker(key))
    warm_up = getattr(reranker, "warm_up", None)
    if warm_up is not None:
        await warm_up()


RerankerDep = Annotated[Reranker, Depends(get_default_reranker)]
