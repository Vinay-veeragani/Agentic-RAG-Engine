import uuid

import pytest
from sqlalchemy import select

from agentic_rag.chunking.base import ChunkingConfig, ChunkingStrategy
from agentic_rag.chunking.pipeline import index_document_version
from agentic_rag.embeddings.providers import MockEmbeddingProvider
from agentic_rag.ingestion.pipeline import ingest_document
from agentic_rag.storage.models import Collection, DocumentChunk


@pytest.fixture
def object_store(tmp_path):
    from agentic_rag.storage.object_store import LocalFileObjectStore

    return LocalFileObjectStore(tmp_path)


@pytest.mark.asyncio
async def test_index_document_version_persists_chunks_with_embeddings(
    db_session, object_store
) -> None:
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()

    content = (
        b"# Overview\n\nRevenue grew 12 percent in 2025.\n\n"
        b"## Risks\n\nSupply chain risk remains elevated for the year.\n"
    )
    result = await ingest_document(
        session=db_session,
        object_store=object_store,
        collection_id=collection.id,
        filename="report.md",
        content=content,
        title=None,
        max_upload_size_bytes=1_000_000,
    )

    config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL, chunk_size_tokens=200)
    chunks = await index_document_version(
        session=db_session,
        document=result.document,
        version=result.version,
        parsed=result.parsed,
        chunking_config=config,
        embedding_provider=MockEmbeddingProvider(),
    )

    assert len(chunks) >= 2
    assert all(c.embedding is not None and len(c.embedding) == 384 for c in chunks)
    assert result.version.status == "indexed"
    assert result.version.chunking_config["strategy"] == "structural"
    assert result.version.embedding_config["provider"] == "mock"

    persisted = await db_session.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id == result.document.id)
    )
    assert len(list(persisted)) == len(chunks)


@pytest.mark.asyncio
async def test_index_document_version_resolves_parent_child_links(db_session, object_store) -> None:
    long_paragraph = ". ".join(f"Detail sentence {i}" for i in range(150))
    content = f"# Big Section\n\n{long_paragraph}\n".encode()

    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()

    result = await ingest_document(
        session=db_session,
        object_store=object_store,
        collection_id=collection.id,
        filename="long.md",
        content=content,
        title=None,
        max_upload_size_bytes=1_000_000,
    )

    config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL, chunk_size_tokens=50)
    chunks = await index_document_version(
        session=db_session,
        document=result.document,
        version=result.version,
        parsed=result.parsed,
        chunking_config=config,
        embedding_provider=MockEmbeddingProvider(),
    )

    parents = [c for c in chunks if c.chunk_metadata.get("is_parent")]
    children = [c for c in chunks if c.parent_chunk_id is not None]
    assert len(parents) == 1
    assert len(children) >= 2
    assert all(c.parent_chunk_id == parents[0].id for c in children)
