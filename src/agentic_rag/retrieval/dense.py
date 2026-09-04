"""Dense (vector similarity) retrieval over pgvector."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.retrieval.base import MetadataFilter, RetrievedCandidate
from agentic_rag.retrieval.filters import build_filter_conditions
from agentic_rag.storage.models import Document, DocumentChunk


class DenseRetriever:
    def __init__(self, session: AsyncSession, embedding_provider: EmbeddingProvider) -> None:
        self._session = session
        self._embeddings = embedding_provider

    async def retrieve(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: MetadataFilter | None = None,
    ) -> list[RetrievedCandidate]:
        [query_vector] = await self._embeddings.embed_texts([query_text])

        # pgvector's cosine_distance is 1 - cosine_similarity for the
        # normalized vectors every provider here produces, so similarity is
        # recovered as `1 - distance` rather than requiring a second query.
        distance = DocumentChunk.embedding.cosine_distance(query_vector)
        stmt = (
            select(DocumentChunk, Document, distance.label("distance"))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.embedding.isnot(None))
            .where(*build_filter_conditions(filters))
            .order_by(distance.asc())
            .limit(top_k)
        )
        rows = (await self._session.execute(stmt)).all()

        candidates: list[RetrievedCandidate] = []
        for chunk, document, dist in rows:
            similarity = 1.0 - float(dist)
            if score_threshold is not None and similarity < score_threshold:
                continue
            candidates.append(
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
                    dense_score=similarity,
                )
            )
        return candidates
