"""Shared retrieval types: the result shape every retriever returns, and the
metadata filter every retriever accepts.

Kept deliberately deterministic and DB-query-shaped — no LLM reasoning
anywhere in this module, per engineering principle #1.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from pydantic import BaseModel

from agentic_rag.core.models import DocumentType


class MetadataFilter(BaseModel):
    """Every field is optional and AND-ed together when present. Restricted
    to a fixed set of columns rather than arbitrary key/value JSON filtering,
    so filters stay "arbitrary" from the caller's point of view but safe;
    safety here comes from a closed field list (no user-supplied column names
    or operators reach SQL), not from sanitizing an open-ended filter
    language."""

    collection_id: uuid.UUID | None = None
    document_type: DocumentType | None = None
    document_ids: list[uuid.UUID] | None = None
    section: str | None = None
    heading: str | None = None
    source: str | None = None
    year: int | None = None  # Document.document_date's year, falling back
    # to created_at (upload time) only for documents with no real date set.


@dataclass(slots=True)
class RetrievedCandidate:
    """One chunk plus its provenance across retrieval methods. Every score
    field is preserved independently rather than collapsed into one number,
    so provenance is never lost. `rank` is 1-indexed position in the
    final (possibly fused) result list."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    page: int | None
    section: str | None
    heading: str | None
    document_filename: str
    document_title: str | None
    document_source: str | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    rank: int | None = None
