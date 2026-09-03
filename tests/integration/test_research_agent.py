import uuid

import pytest

from agentic_rag.agents.research_agent import AgenticRetrievalLoop
from agentic_rag.chunking.base import ChunkingConfig, ChunkingStrategy
from agentic_rag.chunking.pipeline import index_document_version
from agentic_rag.core.config import Settings
from agentic_rag.core.models import TerminationReason
from agentic_rag.embeddings.providers import MockEmbeddingProvider
from agentic_rag.generation.mock import MockLLMProvider
from agentic_rag.ingestion.pipeline import ingest_document
from agentic_rag.retrieval.reranking import MockReranker
from agentic_rag.storage.models import Collection


@pytest.fixture
def object_store(tmp_path):
    from agentic_rag.storage.object_store import LocalFileObjectStore

    return LocalFileObjectStore(tmp_path)


async def _index_text(db_session, object_store, collection_id, filename: str, text: str) -> None:
    result = await ingest_document(
        session=db_session,
        object_store=object_store,
        collection_id=collection_id,
        filename=filename,
        content=text.encode(),
        title=None,
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


def _loop(db_session, *, max_iterations=3, max_calls=8) -> AgenticRetrievalLoop:
    settings = Settings(max_retrieval_iterations=max_iterations, max_retrieval_calls=max_calls)
    return AgenticRetrievalLoop(
        session=db_session,
        llm=MockLLMProvider(),
        embedding_provider=MockEmbeddingProvider(),
        reranker=MockReranker(),
        settings=settings,
    )


@pytest.mark.asyncio
async def test_loop_terminates_with_sufficient_evidence(db_session, object_store) -> None:
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()
    await _index_text(
        db_session,
        object_store,
        collection.id,
        "revenue.txt",
        "Revenue declined due to weaker enterprise demand and pricing pressure.",
    )

    result = await _loop(db_session).run("why did revenue decline", collection_id=collection.id)

    assert result.termination_reason == TerminationReason.SUFFICIENT_EVIDENCE
    assert len(result.iterations) >= 1
    assert result.iterations[-1].sufficient is True
    assert len(result.evidence) >= 1


@pytest.mark.asyncio
async def test_loop_never_exceeds_max_iterations_when_evidence_stays_insufficient(
    db_session, object_store
) -> None:
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()
    await _index_text(
        db_session,
        object_store,
        collection.id,
        "weather.txt",
        "Heavy rainfall is forecast across the region this week.",
    )

    result = await _loop(db_session, max_iterations=2).run(
        "why did quarterly revenue decline", collection_id=collection.id
    )

    assert len(result.iterations) <= 2
    assert result.termination_reason == TerminationReason.MAX_ITERATIONS_REACHED
    assert all(not it.sufficient for it in result.iterations)


@pytest.mark.asyncio
async def test_loop_terminates_with_no_evidence_for_empty_collection(db_session) -> None:
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()

    result = await _loop(db_session).run("anything at all", collection_id=collection.id)

    assert result.termination_reason == TerminationReason.NO_EVIDENCE_FOUND
    assert result.evidence == []
    assert len(result.iterations) == 1


@pytest.mark.asyncio
async def test_loop_scopes_retrieval_to_the_given_collection(db_session, object_store) -> None:
    collection_a = Collection(name=f"col-a-{uuid.uuid4().hex[:8]}")
    collection_b = Collection(name=f"col-b-{uuid.uuid4().hex[:8]}")
    db_session.add_all([collection_a, collection_b])
    await db_session.flush()

    await _index_text(
        db_session, object_store, collection_a.id, "a.txt", "Revenue declined due to demand."
    )

    result = await _loop(db_session).run("why did revenue decline", collection_id=collection_b.id)

    assert result.termination_reason == TerminationReason.NO_EVIDENCE_FOUND


@pytest.mark.asyncio
async def test_loop_respects_max_retrieval_calls_budget(db_session, object_store) -> None:
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()
    await _index_text(
        db_session,
        object_store,
        collection.id,
        "weather.txt",
        "Heavy rainfall is forecast across the region this week.",
    )

    result = await _loop(db_session, max_iterations=5, max_calls=1).run(
        "why did quarterly revenue decline", collection_id=collection.id
    )

    # One iteration consumes the single allowed retrieval call, so the loop
    # must stop before a second retrieval happens.
    assert len(result.iterations) == 1
    assert result.termination_reason == TerminationReason.MAX_RETRIEVAL_CALLS_REACHED
