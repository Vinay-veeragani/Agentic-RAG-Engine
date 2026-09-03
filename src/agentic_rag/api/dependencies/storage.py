from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from agentic_rag.core.config import Settings, get_settings
from agentic_rag.storage.object_store import LocalFileObjectStore, ObjectStore


@lru_cache
def _local_object_store(root_dir: str) -> LocalFileObjectStore:
    return LocalFileObjectStore(root_dir)


def get_object_store(settings: Annotated[Settings, Depends(get_settings)]) -> ObjectStore:
    return _local_object_store(settings.object_store_root)


ObjectStoreDep = Annotated[ObjectStore, Depends(get_object_store)]
