"""Dependency-chained multi-hop retrieval (agents/multi_hop.py).

Plain query decomposition retrieves each subquery independently and fuses
the rankings — it cannot answer a real multi-hop question where the second
hop's evidence is only findable via an entity the *first* hop's evidence
reveals, one the original query never names. This is proven two ways:

1. The naive path — searching for the second hop's literal subquery text
   alone — genuinely does not find the answer (not just "isn't tested"),
   using real Postgres full-text search, not mocked out.
2. The chained path — extracting the bridging entity from hop one's
   evidence and folding it into hop two's query before searching — does
   find it, end to end through the real AgenticRetrievalLoop.

Uses sparse (keyword) retrieval deliberately, forced via a thin LLM
wrapper: MockEmbeddingProvider's embeddings are random-hash based, not
semantically meaningful, so a hybrid/dense comparison would be too noisy
to prove "the naive path cannot find this" reliably — real full-text
search's all-or-nothing lexeme matching (`@@ tsquery`) gives a
deterministic, unambiguous proof instead."""

import uuid

import pytest

from agentic_rag.agents.multi_hop import MultiHopResolver, resolve_second_hop_query
from agentic_rag.agents.planner import RetrievalPlan
from agentic_rag.agents.research_agent import AgenticRetrievalLoop
from agentic_rag.agents.retrieval_agent import RetrievalAgent
from agentic_rag.chunking.base import ChunkingConfig, ChunkingStrategy
from agentic_rag.chunking.pipeline import index_document_version
from agentic_rag.core.config import Settings
from agentic_rag.core.models import RetrievalStrategy
from agentic_rag.embeddings.providers import MockEmbeddingProvider
from agentic_rag.generation.mock import MockLLMProvider
from agentic_rag.ingestion.pipeline import ingest_document
from agentic_rag.retrieval.base import MetadataFilter
from agentic_rag.retrieval.reranking import MockReranker
from agentic_rag.storage.models import Collection

_FIRST_HOP_QUERY = "Who is the CEO of Acme"
_SECOND_HOP_QUERY_RAW = "is that for them?"


class _ForcedSparseLLM:
    """Wraps MockLLMProvider but forces RetrievalPlan.strategy to SPARSE —
    isolates this test from MockEmbeddingProvider's random-hash dense
    scores, which are not semantically meaningful and would make "the
    naive path cannot find this evidence" nondeterministic. Every other
    structured call (analysis, decomposition, entity extraction) still
    goes through the real MockLLMProvider unchanged."""

    def __init__(self) -> None:
        self._inner = MockLLMProvider()

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    async def complete(self, **kwargs: object) -> str:
        return await self._inner.complete(**kwargs)  # type: ignore[arg-type]

    async def complete_structured(self, *, system_prompt, user_prompt, schema, temperature=0.0):
        result = await self._inner.complete_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            temperature=temperature,
        )
        if schema is RetrievalPlan:
            result.strategy = RetrievalStrategy.SPARSE
        return result


@pytest.fixture
def object_store(tmp_path):
    from agentic_rag.storage.object_store import LocalFileObjectStore

    return LocalFileObjectStore(tmp_path)


async def _index(db_session, object_store, collection_id, filename, content):
    result = await ingest_document(
        session=db_session,
        object_store=object_store,
        collection_id=collection_id,
        filename=filename,
        content=content,
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


async def _seed_two_hop_corpus(db_session, object_store) -> uuid.UUID:
    collection = Collection(name=f"col-{uuid.uuid4().hex[:8]}")
    db_session.add(collection)
    await db_session.flush()
    # The second hop's raw text is deliberately all stopwords (Postgres's
    # 'english' full-text config drops every one of its words) — alone it
    # produces an empty tsquery that matches nothing at all. Only the
    # entity extracted from the first hop's evidence ("Jordan Smith")
    # gives the resolved second-hop query any real, matchable content.
    await _index(
        db_session, object_store, collection.id, "ceo.txt", b"The CEO of Acme is Jordan Smith."
    )
    await _index(db_session, object_store, collection.id, "birth.txt", b"Jordan Smith: 1975.")
    return collection.id


@pytest.mark.asyncio
async def test_naive_independent_subquery_search_cannot_find_the_second_hop_evidence(
    db_session, object_store
) -> None:
    """The control case: proves the gap chaining closes actually exists.
    Searching for the second hop's subquery text alone, with no entity
    substitution, must not find "birth.txt" — real Postgres full-text
    search, not an assumption."""
    collection_id = await _seed_two_hop_corpus(db_session, object_store)
    plan = RetrievalPlan(
        strategy=RetrievalStrategy.SPARSE,
        filters=MetadataFilter(collection_id=collection_id),
    )
    agent = RetrievalAgent(db_session, MockEmbeddingProvider(), MockReranker())

    outcome = await agent.retrieve([_SECOND_HOP_QUERY_RAW], plan)

    assert all(c.document_filename != "birth.txt" for c in outcome.candidates)


@pytest.mark.asyncio
async def test_chained_multi_hop_retrieval_finds_evidence_naive_search_cannot(
    db_session, object_store
) -> None:
    """The real fix: run the full agentic loop over a genuine multi-hop
    query and confirm it actually retrieves "birth.txt" — evidence the
    control case above proves is unreachable without the entity bridge."""
    collection_id = await _seed_two_hop_corpus(db_session, object_store)

    settings = Settings(max_retrieval_iterations=1, max_retrieval_calls=8)
    loop = AgenticRetrievalLoop(
        session=db_session,
        llm=_ForcedSparseLLM(),
        embedding_provider=MockEmbeddingProvider(),
        reranker=MockReranker(),
        settings=settings,
    )

    query = f"{_FIRST_HOP_QUERY}, and {_SECOND_HOP_QUERY_RAW}"
    result = await loop.run(query, collection_id=collection_id)

    assert len(result.iterations) == 1
    queries_used = result.iterations[0].queries_used
    assert len(queries_used) == 2
    # The second hop's *resolved* query must carry the entity extracted
    # from the first hop's evidence — proof the chain actually happened,
    # not just that two retrieval calls were made independently.
    assert "Jordan Smith" in queries_used[1]
    assert any(c.document_filename == "birth.txt" for c in result.evidence)


@pytest.mark.asyncio
async def test_resolve_second_hop_query_folds_entity_into_the_query() -> None:
    assert resolve_second_hop_query("what year", "Jordan Smith") == "what year Jordan Smith"
    assert resolve_second_hop_query("what year", "") == "what year"


@pytest.mark.asyncio
async def test_multi_hop_resolver_extracts_an_entity_from_evidence(
    db_session, object_store
) -> None:
    from agentic_rag.retrieval.dense import DenseRetriever

    collection_id = await _seed_two_hop_corpus(db_session, object_store)
    dense = DenseRetriever(db_session, MockEmbeddingProvider())
    candidates = await dense.retrieve(
        "CEO", top_k=10, filters=MetadataFilter(collection_id=collection_id)
    )
    ceo_evidence = [c for c in candidates if c.document_filename == "ceo.txt"]

    resolver = MultiHopResolver(MockLLMProvider())
    entity = await resolver.extract_bridge_entity(_FIRST_HOP_QUERY, ceo_evidence)

    assert entity == "Jordan Smith"


@pytest.mark.asyncio
async def test_multi_hop_resolver_returns_empty_string_for_no_evidence() -> None:
    resolver = MultiHopResolver(MockLLMProvider())
    entity = await resolver.extract_bridge_entity("anything", [])
    assert entity == ""
