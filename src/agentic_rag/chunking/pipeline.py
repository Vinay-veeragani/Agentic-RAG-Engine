"""Chunk + embed + persist: turns a parsed document into indexed
`DocumentChunk` rows (spec §7/§8/§9 "indexing")."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.chunking.base import ChunkingConfig, get_chunker
from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.ingestion.parsed_document import ParsedDocument
from agentic_rag.storage.models import Document, DocumentChunk, DocumentVersion


async def index_document_version(
    *,
    session: AsyncSession,
    document: Document,
    version: DocumentVersion,
    parsed: ParsedDocument,
    chunking_config: ChunkingConfig,
    embedding_provider: EmbeddingProvider,
) -> list[DocumentChunk]:
    chunker = get_chunker(chunking_config.strategy, embedding_provider=embedding_provider)
    candidates = await chunker.chunk(parsed, chunking_config)

    version.chunking_config = chunking_config.model_dump(mode="json")
    version.embedding_config = {
        "provider": embedding_provider.model_name,
        "dimensions": embedding_provider.dimensions,
    }

    if not candidates:
        version.status = "indexed"
        await session.flush()
        return []

    embeddings = await embedding_provider.embed_texts([c.text for c in candidates])

    created: list[DocumentChunk] = []
    for candidate, embedding in zip(candidates, embeddings, strict=True):
        created.append(
            DocumentChunk(
                document_version_id=version.id,
                document_id=document.id,
                chunk_index=candidate.order_index,
                page=candidate.page,
                section=candidate.section,
                heading=candidate.heading,
                content=candidate.text,
                token_count=candidate.token_count,
                character_count=candidate.character_count,
                embedding=embedding,
                chunk_metadata=candidate.metadata,
            )
        )

    # SQLAlchemy applies a column's Python-side `default=` callable at flush
    # time, not at object construction — `chunk.id` is None until this flush
    # actually runs, so parent_chunk_id can only be resolved afterward.
    session.add_all(created)
    await session.flush()

    for candidate, chunk in zip(candidates, created, strict=True):
        if candidate.parent_index is not None:
            chunk.parent_chunk_id = created[candidate.parent_index].id

    version.status = "indexed"
    await session.flush()
    return created
