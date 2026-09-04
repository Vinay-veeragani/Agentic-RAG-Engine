import uuid

import pytest
from sqlalchemy import select

from agentic_rag.ingestion.pipeline import ingest_document
from agentic_rag.storage.models import Collection, DocumentVersion


@pytest.fixture
def object_store(tmp_path):
    from agentic_rag.storage.object_store import LocalFileObjectStore

    return LocalFileObjectStore(tmp_path)


@pytest.mark.asyncio
async def test_ingest_document_creates_document_and_version(db_session, object_store) -> None:
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()

    result = await ingest_document(
        session=db_session,
        object_store=object_store,
        collection_id=collection.id,
        filename="notes.txt",
        content=b"Hello world.\n\nSecond paragraph.",
        title=None,
        max_upload_size_bytes=1_000_000,
    )

    assert result.document.filename == "notes.txt"
    assert result.document.document_type == "txt"
    assert result.version.version_number == 1
    assert len(result.parsed.elements) == 2

    stored = await object_store.load(result.version.storage_path)
    assert stored == b"Hello world.\n\nSecond paragraph."


@pytest.mark.asyncio
async def test_ingest_document_reuploading_same_filename_creates_new_version(
    db_session, object_store
) -> None:
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()

    first = await ingest_document(
        session=db_session,
        object_store=object_store,
        collection_id=collection.id,
        filename="report.txt",
        content=b"version one content",
        title=None,
        max_upload_size_bytes=1_000_000,
    )
    second = await ingest_document(
        session=db_session,
        object_store=object_store,
        collection_id=collection.id,
        filename="report.txt",
        content=b"version two content, different",
        title=None,
        max_upload_size_bytes=1_000_000,
    )

    assert first.document.id == second.document.id
    assert first.version.version_number == 1
    assert second.version.version_number == 2

    versions = await db_session.scalars(
        select(DocumentVersion).where(DocumentVersion.document_id == first.document.id)
    )
    assert len(list(versions)) == 2


@pytest.mark.asyncio
async def test_ingest_document_reuploading_identical_content_is_a_no_op(
    db_session, object_store
) -> None:
    """A byte-identical re-upload must not create a redundant version —
    the checksum this module computes is actually checked against
    something, not just stored and ignored."""
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()

    content = b"identical content, uploaded twice"
    first = await ingest_document(
        session=db_session,
        object_store=object_store,
        collection_id=collection.id,
        filename="report.txt",
        content=content,
        title=None,
        max_upload_size_bytes=1_000_000,
    )
    second = await ingest_document(
        session=db_session,
        object_store=object_store,
        collection_id=collection.id,
        filename="report.txt",
        content=content,
        title=None,
        max_upload_size_bytes=1_000_000,
    )

    assert first.document.id == second.document.id
    assert first.version.id == second.version.id
    assert second.version.version_number == 1

    versions = await db_session.scalars(
        select(DocumentVersion).where(DocumentVersion.document_id == first.document.id)
    )
    assert len(list(versions)) == 1
