"""Metadata-filter SQL condition building, shared by every retriever, plus
`MetadataRetriever` — filter-only retrieval with no relevance ranking (spec
§9's fourth retriever: sometimes you want "every chunk in section X", not a
similarity search).
"""

from __future__ import annotations

from sqlalchemy import ColumnElement, extract, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.retrieval.base import MetadataFilter, RetrievedCandidate
from agentic_rag.storage.models import Document, DocumentChunk


def build_filter_conditions(filters: MetadataFilter | None) -> list[ColumnElement[bool]]:
    """Assumes the query already joins DocumentChunk to Document."""
    if filters is None:
        return []

    conditions: list[ColumnElement[bool]] = []
    if filters.collection_id is not None:
        conditions.append(Document.collection_id == filters.collection_id)
    if filters.document_type is not None:
        conditions.append(Document.document_type == filters.document_type.value)
    if filters.document_ids is not None:
        conditions.append(DocumentChunk.document_id.in_(filters.document_ids))
    if filters.section is not None:
        conditions.append(DocumentChunk.section == filters.section)
    if filters.heading is not None:
        conditions.append(DocumentChunk.heading == filters.heading)
    if filters.source is not None:
        conditions.append(Document.source == filters.source)
    if filters.year is not None:
        # Prefer the document's real-world date when the caller supplied one
        # at upload time (Document.document_date); created_at is upload
        # time, not document content, and is only a fallback proxy for
        # documents where no real date was ever provided.
        conditions.append(
            extract(
                "year", func.coalesce(Document.document_date, Document.created_at)
            )
            == filters.year
        )
    return conditions


def _row_to_candidate(chunk: DocumentChunk, document: Document) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        content=chunk.content,
        page=chunk.page,
        section=chunk.section,
        heading=chunk.heading,
        document_filename=document.filename,
        document_title=document.title,
        document_source=document.source,
    )


class MetadataRetriever:
    """Returns chunks matching `filters` only, ordered by document recency
    then chunk order — no query text, no relevance scoring."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def retrieve(
        self, *, filters: MetadataFilter, top_k: int = 50
    ) -> list[RetrievedCandidate]:
        stmt = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(*build_filter_conditions(filters))
            .order_by(Document.created_at.desc(), DocumentChunk.chunk_index.asc())
            .limit(top_k)
        )
        rows = (await self._session.execute(stmt)).all()
        return [_row_to_candidate(chunk, document) for chunk, document in rows]
