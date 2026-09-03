"""Deterministic metadata derived from raw upload bytes, independent of parsing.

Parsed-content metadata (title, page count, per-element metadata) lives on
`ParsedDocument` instead — this module only covers what can be computed
without understanding the file format at all.
"""

from __future__ import annotations

import hashlib


def compute_checksum(content: bytes) -> str:
    """SHA-256 hex digest — used for the documents/document_versions checksum
    columns (spec §6/§28) and for cheap duplicate-version detection."""
    return hashlib.sha256(content).hexdigest()
