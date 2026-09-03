"""Sparse (PostgreSQL full-text) retrieval (spec §9).

Uses the `content_tsv` generated column (see storage/models.py) and
`ts_rank_cd`, which — unlike `ts_rank` — accounts for lexeme proximity, so it
rewards a phrase-like match over the same terms scattered far apart. This is
Postgres full-text search, not literal BM25 — see docs/architecture.md for
why that distinction is called out rather than glossed over.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.retrieval.base import MetadataFilter, RetrievedCandidate
from agentic_rag.retrieval.filters import build_filter_conditions
from agentic_rag.storage.models import Document, DocumentChunk


class SparseRetriever:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def retrieve(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        filters: MetadataFilter | None = None,
    ) -> list[RetrievedCandidate]:
        tsquery = func.plainto_tsquery("english", query_text)
        rank = func.ts_rank_cd(DocumentChunk.content_tsv, tsquery)

        stmt = (
            select(DocumentChunk, Document, rank.label("rank"))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.content_tsv.op("@@")(tsquery))
            .where(*build_filter_conditions(filters))
            .order_by(rank.desc())
            .limit(top_k)
        )
        rows = (await self._session.execute(stmt)).all()

        return [
            RetrievedCandidate(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                page=chunk.page,
                section=chunk.section,
                heading=chunk.heading,
                document_filename=document.filename,
                document_title=document.title,
                document_source=document.source,
                sparse_score=float(rank_value),
            )
            for chunk, document, rank_value in rows
        ]
