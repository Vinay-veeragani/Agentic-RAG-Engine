from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from agentic_rag.core.config import Settings, get_settings
from agentic_rag.generation.llm import LLMProvider
from agentic_rag.generation.providers import get_llm_provider

_provider_instances: dict[str, LLMProvider] = {}


def get_default_llm_provider(settings: Annotated[Settings, Depends(get_settings)]) -> LLMProvider:
    key = settings.llm_provider
    if key not in _provider_instances:
        _provider_instances[key] = get_llm_provider(settings.llm_provider, settings)
    return _provider_instances[key]


LLMProviderDep = Annotated[LLMProvider, Depends(get_default_llm_provider)]
