"""MetadataFilter.year previously filtered on Document.created_at (upload
time) unconditionally — a document about fiscal year 2020 uploaded today
would filter under the current year, not 2020. Found during an engineering
audit. This proves the real fix: a caller-supplied document_date is
preferred, with created_at only as a fallback for documents where no real
date was ever provided."""

import datetime
import uuid

import pytest

from agentic_rag.chunking.base import ChunkingConfig, ChunkingStrategy
from agentic_rag.chunking.pipeline import index_document_version
from agentic_rag.embeddings.providers import MockEmbeddingProvider
from agentic_rag.ingestion.pipeline import ingest_document
from agentic_rag.retrieval.base import MetadataFilter
from agentic_rag.retrieval.filters import MetadataRetriever
from agentic_rag.storage.models import Collection


@pytest.fixture
def object_store(tmp_path):
    from agentic_rag.storage.object_store import LocalFileObjectStore

    return LocalFileObjectStore(tmp_path)


async def _index(db_session, object_store, collection_id, filename, content, *, document_date=None):
    result = await ingest_document(
        session=db_session,
        object_store=object_store,
        collection_id=collection_id,
        filename=filename,
        content=content,
        title=None,
        document_date=document_date,
        max_upload_size_bytes=1_000_000,
    )
    config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL, chunk_size_tokens=100)
    await index_document_version(
        session=db_session,
        document=result.document,
        version=result.version,
        parsed=result.parsed,
        chunking_config=config,
        embedding_provider=MockEmbeddingProvider(),
    )


@pytest.mark.asyncio
async def test_year_filter_prefers_document_date_over_upload_time(
    db_session, object_store
) -> None:
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()

    # Uploaded "today" (created_at falls in the current year), but its real
    # content is from fiscal year 2020.
    await _index(
        db_session,
        object_store,
        collection.id,
        "fy2020.txt",
        b"Fiscal year 2020 results.",
        document_date=datetime.date(2020, 6, 30),
    )

    matches_2020 = await MetadataRetriever(db_session).retrieve(
        filters=MetadataFilter(collection_id=collection.id, year=2020)
    )
    assert len(matches_2020) == 1

    matches_current_year = await MetadataRetriever(db_session).retrieve(
        filters=MetadataFilter(collection_id=collection.id, year=datetime.date.today().year)
    )
    assert matches_current_year == []


@pytest.mark.asyncio
async def test_year_filter_falls_back_to_created_at_when_no_document_date(
    db_session, object_store
) -> None:
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()

    await _index(
        db_session,
        object_store,
        collection.id,
        "undated.txt",
        b"No explicit document date was ever provided for this one.",
    )

    matches = await MetadataRetriever(db_session).retrieve(
        filters=MetadataFilter(collection_id=collection.id, year=datetime.date.today().year)
    )
    assert len(matches) == 1
