"""Schema-level guardrails against duplicate rows (found missing during an
engineering audit): the application layer should never produce a duplicate
chunk_index within a document version, or record the same chunk twice for
one retrieval run — but until these constraints existed, nothing at the
database level would have stopped it either."""

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from agentic_rag.storage.models import (
    Collection,
    Document,
    DocumentChunk,
    DocumentVersion,
    Query,
    RetrievalRun,
    RetrievedChunk,
)


@pytest.mark.asyncio
async def test_document_chunks_reject_duplicate_chunk_index_within_a_version(
    db_session,
) -> None:
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    document = Document(
        collection=collection,
        filename="a.txt",
        document_type="txt",
        checksum="checksum-a",
    )
    version = DocumentVersion(
        document=document, version_number=1, checksum="checksum-a", storage_path="a.txt"
    )
    db_session.add_all([collection, document, version])
    await db_session.flush()

    db_session.add(
        DocumentChunk(
            document_version_id=version.id,
            document_id=document.id,
            chunk_index=0,
            content="first chunk",
            token_count=2,
            character_count=11,
        )
    )
    await db_session.flush()

    db_session.add(
        DocumentChunk(
            document_version_id=version.id,
            document_id=document.id,
            chunk_index=0,
            content="duplicate index",
            token_count=2,
            character_count=16,
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_retrieved_chunks_reject_the_same_chunk_recorded_twice_for_one_run(
    db_session,
) -> None:
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    document = Document(
        collection=collection,
        filename="a.txt",
        document_type="txt",
        checksum="checksum-a",
    )
    version = DocumentVersion(
        document=document, version_number=1, checksum="checksum-a", storage_path="a.txt"
    )
    query = Query(trace_id=uuid.uuid4().hex, query_text="why did revenue decline")
    db_session.add_all([collection, document, version, query])
    await db_session.flush()

    chunk = DocumentChunk(
        document_version_id=version.id,
        document_id=document.id,
        chunk_index=0,
        content="revenue declined",
        token_count=2,
        character_count=17,
    )
    run = RetrievalRun(
        query_id=query.id,
        iteration_number=1,
        retrieval_strategy="hybrid",
        query_text_used="why did revenue decline",
    )
    db_session.add_all([chunk, run])
    await db_session.flush()

    db_session.add(RetrievedChunk(retrieval_run_id=run.id, chunk_id=chunk.id, rank=1))
    await db_session.flush()

    db_session.add(RetrievedChunk(retrieval_run_id=run.id, chunk_id=chunk.id, rank=2))
    with pytest.raises(IntegrityError):
        await db_session.flush()
