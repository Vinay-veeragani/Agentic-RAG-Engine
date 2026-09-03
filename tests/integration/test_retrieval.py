import uuid

import pytest
import pytest_asyncio

from agentic_rag.chunking.base import ChunkingConfig, ChunkingStrategy
from agentic_rag.chunking.pipeline import index_document_version
from agentic_rag.embeddings.providers import MockEmbeddingProvider
from agentic_rag.ingestion.pipeline import ingest_document
from agentic_rag.retrieval.base import MetadataFilter
from agentic_rag.retrieval.dense import DenseRetriever
from agentic_rag.retrieval.filters import MetadataRetriever
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.sparse import SparseRetriever
from agentic_rag.storage.models import Collection


@pytest.fixture
def object_store(tmp_path):
    from agentic_rag.storage.object_store import LocalFileObjectStore

    return LocalFileObjectStore(tmp_path)


async def _index_text(
    db_session, object_store, collection_id, filename: str, text: str, *, source: str | None = None
) -> None:
    result = await ingest_document(
        session=db_session,
        object_store=object_store,
        collection_id=collection_id,
        filename=filename,
        content=text.encode(),
        title=None,
        source=source,
        max_upload_size_bytes=1_000_000,
    )
    config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL, chunk_size_tokens=200)
    await index_document_version(
        session=db_session,
        document=result.document,
        version=result.version,
        parsed=result.parsed,
        chunking_config=config,
        embedding_provider=MockEmbeddingProvider(),
    )


@pytest_asyncio.fixture
async def indexed_collection(db_session, object_store) -> uuid.UUID:
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()

    await _index_text(
        db_session,
        object_store,
        collection.id,
        "revenue.txt",
        "Quarterly revenue increased significantly due to strong product demand.",
    )
    await _index_text(
        db_session,
        object_store,
        collection.id,
        "weather.txt",
        "The weather forecast predicts heavy rainfall across the region this week.",
    )
    return collection.id


@pytest.mark.asyncio
async def test_sparse_retriever_finds_exact_term_match(db_session, indexed_collection) -> None:
    retriever = SparseRetriever(db_session)
    results = await retriever.retrieve(
        "revenue", top_k=5, filters=MetadataFilter(collection_id=indexed_collection)
    )

    assert len(results) >= 1
    assert "revenue" in results[0].content.lower()
    assert results[0].sparse_score is not None and results[0].sparse_score > 0


@pytest.mark.asyncio
async def test_sparse_retriever_returns_empty_for_no_match(db_session, indexed_collection) -> None:
    retriever = SparseRetriever(db_session)
    results = await retriever.retrieve(
        "nonexistent_term_xyz", top_k=5, filters=MetadataFilter(collection_id=indexed_collection)
    )
    assert results == []


@pytest.mark.asyncio
async def test_dense_retriever_returns_scored_and_sorted_results(
    db_session, indexed_collection
) -> None:
    retriever = DenseRetriever(db_session, MockEmbeddingProvider())
    results = await retriever.retrieve(
        "anything", top_k=5, filters=MetadataFilter(collection_id=indexed_collection)
    )

    assert len(results) >= 1
    scores = [r.dense_score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(r.dense_score is not None for r in results)


@pytest.mark.asyncio
async def test_hybrid_retriever_assigns_fusion_score_and_rank(
    db_session, indexed_collection
) -> None:
    retriever = HybridRetriever(db_session, MockEmbeddingProvider())
    results = await retriever.retrieve(
        "revenue", top_k=5, filters=MetadataFilter(collection_id=indexed_collection)
    )

    assert len(results) >= 1
    assert results[0].rank == 1
    assert all(r.fusion_score is not None for r in results)
    ranks = [r.rank for r in results]
    assert ranks == sorted(ranks)


@pytest.mark.asyncio
async def test_hybrid_retriever_preserves_document_source(db_session, object_store) -> None:
    """Regression test: HybridRetriever's fusion step used to rebuild each
    RetrievedCandidate without copying document_source, silently dropping it
    even though DenseRetriever/SparseRetriever both populated it — found via
    an end-to-end source-authority test, not a targeted unit test."""
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()
    await _index_text(
        db_session,
        object_store,
        collection.id,
        "annual.txt",
        "Quarterly revenue increased significantly.",
        source="Annual Report",
    )

    retriever = HybridRetriever(db_session, MockEmbeddingProvider())
    results = await retriever.retrieve(
        "revenue", top_k=5, filters=MetadataFilter(collection_id=collection.id)
    )

    assert len(results) >= 1
    assert results[0].document_source == "Annual Report"


@pytest.mark.asyncio
async def test_hybrid_retriever_respects_metadata_filters(db_session, indexed_collection) -> None:
    other_collection_id = uuid.uuid4()
    retriever = HybridRetriever(db_session, MockEmbeddingProvider())
    results = await retriever.retrieve(
        "revenue", top_k=5, filters=MetadataFilter(collection_id=other_collection_id)
    )
    assert results == []


@pytest.mark.asyncio
async def test_metadata_retriever_filters_without_a_query(db_session, indexed_collection) -> None:
    retriever = MetadataRetriever(db_session)
    results = await retriever.retrieve(
        filters=MetadataFilter(collection_id=indexed_collection), top_k=10
    )
    assert len(results) >= 2
