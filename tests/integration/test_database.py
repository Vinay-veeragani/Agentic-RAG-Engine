import uuid

import pytest
from sqlalchemy import select

from agentic_rag.storage.models import Collection


@pytest.mark.asyncio
async def test_collection_round_trip(db_session) -> None:
    name = f"test-collection-{uuid.uuid4().hex[:8]}"
    collection = Collection(name=name, description="integration test collection")
    db_session.add(collection)
    await db_session.flush()

    fetched = await db_session.scalar(select(Collection).where(Collection.name == name))
    assert fetched is not None
    assert fetched.id == collection.id
    assert fetched.retrieval_config == {}


@pytest.mark.asyncio
async def test_document_chunk_embedding_round_trip(db_session) -> None:
    """Confirms the pgvector column actually stores/returns a vector, not just
    that the migration created the column."""
    from agentic_rag.storage.models import Document, DocumentChunk, DocumentVersion

    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()

    document = Document(
        collection_id=collection.id,
        filename="smoke.txt",
        document_type="txt",
        checksum="deadbeef",
    )
    db_session.add(document)
    await db_session.flush()

    version = DocumentVersion(
        document_id=document.id,
        version_number=1,
        checksum="deadbeef",
        storage_path="/tmp/smoke.txt",
    )
    db_session.add(version)
    await db_session.flush()

    embedding = [0.1] * 384
    chunk = DocumentChunk(
        document_version_id=version.id,
        document_id=document.id,
        chunk_index=0,
        content="hello world",
        token_count=2,
        character_count=11,
        embedding=embedding,
    )
    db_session.add(chunk)
    await db_session.flush()

    fetched = await db_session.get(DocumentChunk, chunk.id)
    assert fetched is not None
    assert fetched.embedding is not None
    assert len(fetched.embedding) == 384
