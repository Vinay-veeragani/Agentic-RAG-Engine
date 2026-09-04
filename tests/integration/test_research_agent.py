import uuid

import pytest

from agentic_rag.agents.evidence_agent import EvidenceAssessment
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


class _ForcedRefinementLLM:
    """Wraps MockLLMProvider but forces exactly one insufficient evidence
    judgment before switching to sufficient, so a test can prove the loop's
    "insufficient -> refine -> retry -> sufficient" transition actually runs
    end to end, with a real second retrieval call using a genuinely refined
    query — not just the two extremes (instant success, permanent failure)
    every other test in this file covers. Every other structured call
    (query analysis, planning, expansion, decomposition) still goes through
    the real MockLLMProvider, unchanged."""

    def __init__(self) -> None:
        self._inner = MockLLMProvider()
        self._evidence_calls = 0

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    async def complete(self, **kwargs: object) -> str:
        return await self._inner.complete(**kwargs)  # type: ignore[arg-type]

    async def complete_structured(self, *, system_prompt, user_prompt, schema, temperature=0.0):
        if schema is EvidenceAssessment:
            self._evidence_calls += 1
            if self._evidence_calls == 1:
                return EvidenceAssessment(
                    sufficient=False,
                    reason="Evidence confirms what happened but not why.",
                    missing_information=["the specific cause of the decline"],
                    relevance=0.6,
                    coverage=0.4,
                    directness=0.3,
                )
            return EvidenceAssessment(
                sufficient=True,
                reason="Evidence now explains the cause.",
                missing_information=[],
                relevance=0.9,
                coverage=0.9,
                directness=0.9,
            )
        return await self._inner.complete_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            temperature=temperature,
        )


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
async def test_loop_refines_query_and_succeeds_on_second_iteration(
    db_session, object_store
) -> None:
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

    llm = _ForcedRefinementLLM()
    settings = Settings(max_retrieval_iterations=3, max_retrieval_calls=8)
    loop = AgenticRetrievalLoop(
        session=db_session,
        llm=llm,
        embedding_provider=MockEmbeddingProvider(),
        reranker=MockReranker(),
        settings=settings,
    )

    result = await loop.run("why did revenue decline", collection_id=collection.id)

    assert result.termination_reason == TerminationReason.SUFFICIENT_EVIDENCE
    assert len(result.iterations) == 2
    assert result.iterations[0].sufficient is False
    assert result.iterations[1].sufficient is True
    # The second retrieval call genuinely used a different, refined query —
    # this is a real retry with new input, not a no-op re-run.
    assert result.iterations[1].queries_used != result.iterations[0].queries_used
    assert "cause of the decline" in result.iterations[1].queries_used[0]
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


@pytest.mark.asyncio
async def test_loop_terminates_with_conflicting_evidence_when_unresolved(
    db_session, object_store
) -> None:
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()

    await _index_text(
        db_session, object_store, collection.id, "a.txt", "Revenue declined 4% in Q3."
    )
    await _index_text(
        db_session, object_store, collection.id, "b.txt", "Revenue declined 9% in Q3."
    )

    result = await _loop(db_session).run("what happened to revenue", collection_id=collection.id)

    assert result.termination_reason == TerminationReason.CONFLICTING_EVIDENCE
    assert len(result.iterations) == 1
    assert result.iterations[0].contradictions
    assert result.iterations[0].contradictions[0].resolution is None


@pytest.mark.asyncio
async def test_loop_does_not_stop_on_conflict_resolved_by_source_authority(
    db_session, object_store
) -> None:
    collection = Collection(
        name=f"col-{uuid.uuid4().hex[:8]}",
        source_authority_config={"order": ["annual report", "press release"]},
    )
    db_session.add(collection)
    await db_session.flush()

    await _index_text(
        db_session,
        object_store,
        collection.id,
        "annual.txt",
        "Revenue declined 4% in Q3.",
        source="Annual Report",
    )
    await _index_text(
        db_session,
        object_store,
        collection.id,
        "press.txt",
        "Revenue declined 9% in Q3.",
        source="Press Release",
    )

    result = await _loop(db_session).run("what happened to revenue", collection_id=collection.id)

    assert result.termination_reason != TerminationReason.CONFLICTING_EVIDENCE
    assert result.iterations[0].contradictions
    assert result.iterations[0].contradictions[0].resolution is not None
