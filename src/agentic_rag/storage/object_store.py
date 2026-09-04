"""Object storage interface + a local-filesystem implementation.

Raw ingested files (the PDF/DOCX/etc a user uploads) are stored here, keyed by
document/version, separate from the parsed+chunked representation in Postgres.
Local filesystem is the only implementation for now — swapping to S3-compatible
storage later is a matter of adding a second class behind this Protocol, not a
rewrite of ingestion (spec principle: replaceable backends, no unneeded
microservices/infra until something concrete needs it).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class ObjectStore(Protocol):
    async def save(self, key: str, data: bytes) -> str:
        """Persist `data` under `key`; returns the storage key/reference to
        persist (e.g. as `DocumentVersion.storage_path`) and pass back into
        `load()`/`delete()` later — NOT necessarily a raw filesystem path;
        callers must treat the return value as an opaque reference."""
        ...

    async def load(self, key: str) -> bytes: ...

    async def delete(self, key: str) -> None: ...


class LocalFileObjectStore:
    """Stores objects under a root directory, one file per key.

    `key` must not escape `root_dir` — rejects any key that resolves outside
    it (path traversal protection).
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        path = (self.root_dir / key).resolve()
        if os.path.commonpath([path, self.root_dir]) != str(self.root_dir):
            raise ValueError(f"object key escapes storage root: {key!r}")
        return path

    async def save(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    async def load(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    async def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            path.unlink()
