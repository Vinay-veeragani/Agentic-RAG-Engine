"""A small, self-contained benchmark corpus + case set (covering the
standard benchmark category list) built fresh at evaluation time — not a hardcoded set of
"expected answers," but real documents ingested through the real pipeline,
with ground truth (`relevant_document_ids`) resolved from the real
`Document` rows the ingestion step actually created. Recall/precision/etc.
computed against this are real numbers from a real (if small and
synthetic) corpus, not fabricated.

Document-level (not chunk-level) relevance ground truth: a retrieved chunk
counts as relevant if it belongs to one of the case's relevant documents.
Chunk-level ground truth would be more precise but depends on exactly how a
document gets split, which varies with chunking config — document identity
does not.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.chunking.base import ChunkingConfig, ChunkingStrategy
from agentic_rag.chunking.pipeline import index_document_version
from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.ingestion.pipeline import ingest_document
from agentic_rag.storage.models import Collection
from agentic_rag.storage.object_store import ObjectStore


class BenchmarkCategory(StrEnum):
    SIMPLE_FACTUAL = "simple_factual"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    ANALYTICAL = "analytical"
    AGGREGATION = "aggregation"
    AMBIGUOUS = "ambiguous"
    UNANSWERABLE = "unanswerable"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"


@dataclass(slots=True, frozen=True)
class _CorpusDocument:
    filename: str
    text: str
    source: str | None = None


@dataclass(slots=True, frozen=True)
class EvalCase:
    query: str
    category: BenchmarkCategory
    relevant_filenames: tuple[str, ...] = ()
    expected_keywords: tuple[str, ...] = ()


@dataclass(slots=True)
class BuiltEvalCase:
    query: str
    category: BenchmarkCategory
    relevant_document_ids: set[uuid.UUID]
    expected_keywords: tuple[str, ...] = field(default_factory=tuple)


_CORPUS_DOCUMENTS: tuple[_CorpusDocument, ...] = (
    _CorpusDocument(
        "revenue_2023.txt",
        "In fiscal year 2023, the company reported total revenue of $120 million, "
        "representing an 8 percent increase over the prior year. Operating margin "
        "was 22 percent for the year.",
        source="Annual Report",
    ),
    _CorpusDocument(
        "revenue_2024.txt",
        "In fiscal year 2024, the company reported total revenue of $150 million, "
        "representing a 25 percent increase over the prior year. Operating margin "
        "improved to 25 percent, driven by cost discipline.",
        source="Annual Report",
    ),
    _CorpusDocument(
        "risk_factors.txt",
        "The company faces risks including increased competition, foreign currency "
        "fluctuation, and supply chain disruption. Regulatory changes in key "
        "markets could adversely affect operations and future revenue growth.",
        source="Annual Report",
    ),
    _CorpusDocument(
        "rd_spending.txt",
        "Research and development spending in 2024 totaled $30 million, a 20 "
        "percent increase from $25 million in 2023, reflecting continued "
        "investment in product innovation.",
        source="Annual Report",
    ),
    # Deliberately unclassified sources: the point of this pair is to
    # demonstrate a genuinely *unresolved* conflict (the system should tell
    # the user explicitly rather than silently pick a side), which requires
    # neither the default nor any per-collection authority order being able
    # to distinguish them. Giving them distinct
    # standard labels (e.g. "Annual Report" vs "Press Release") would let the
    # built-in default authority order silently resolve this case instead —
    # a real, correct behavior, just not the one this case exists to show.
    _CorpusDocument(
        "conflicting_revenue_a.txt",
        "Quarterly revenue for Q3 2024 declined 4 percent due to weaker demand.",
    ),
    _CorpusDocument(
        "conflicting_revenue_b.txt",
        "Quarterly revenue for Q3 2024 declined 9 percent due to weaker demand.",
    ),
    _CorpusDocument(
        "weather_unrelated.txt",
        "Heavy rainfall affected shipping operations in the coastal region during "
        "the third quarter, delaying several inbound freight shipments.",
    ),
    # The following distractor documents share no revenue/margin/profit content
    # with the cases above. They exist so the corpus is large enough (relative
    # to the default evidence pool size) that retrieval ranking actually has
    # to discriminate relevant from irrelevant documents — with only the
    # handful of finance documents above, nearly the whole corpus would be
    # retrieved for every query regardless of relevance, and an unrelated
    # contradiction (conflicting_revenue_a/b) would leak into every case's
    # evidence. This was found by actually running the benchmark, not
    # designed in from the start.
    _CorpusDocument(
        "office_relocation.txt",
        "The company announced plans to relocate its headquarters to a new "
        "office campus next year, citing space constraints at the current site.",
    ),
    _CorpusDocument(
        "employee_benefits.txt",
        "The company expanded employee health benefits, adding paid parental "
        "leave and a wellness program subsidy for all full-time staff.",
    ),
    _CorpusDocument(
        "supply_chain_update.txt",
        "The company diversified its supplier base across three additional "
        "countries to reduce dependency on any single manufacturing region.",
    ),
    _CorpusDocument(
        "sustainability_initiative.txt",
        "The company invested in renewable energy at its manufacturing "
        "facilities, reducing carbon emissions from on-site operations.",
    ),
    _CorpusDocument(
        "product_launch.txt",
        "The company launched a new product line targeting small and "
        "medium-sized businesses, expanding beyond its traditional customer base.",
    ),
    _CorpusDocument(
        "customer_support.txt",
        "The company opened a new customer support center to reduce average "
        "response times and improve satisfaction scores for enterprise clients.",
    ),
    _CorpusDocument(
        "data_center_upgrade.txt",
        "The company upgraded its primary data center with additional server "
        "capacity to support growing demand for its cloud-hosted services.",
    ),
    _CorpusDocument(
        "board_appointment.txt",
        "The board of directors appointed a new independent member with "
        "experience in international regulatory compliance.",
    ),
    # A genuine two-hop pair: cfo_2024.txt answers "who", and
    # cfo_prior_role.txt answers "what did they do before" — but only by
    # name, not by any term the multi-hop query below uses. A plain
    # independent-subquery search for the second hop's literal text
    # cannot find cfo_prior_role.txt at all (verified: its own tsquery is
    # empty, an all-stopword phrase); only a real dependency chain that
    # extracts "Morgan Reyes" from the first hop's evidence and folds it
    # into the second hop's query can (see agents/multi_hop.py and
    # tests/integration/test_multi_hop.py, which prove this with the same
    # mechanism against a smaller, isolated corpus).
    _CorpusDocument(
        "cfo_2024.txt",
        "Morgan Reyes was the CFO during fiscal year 2024, overseeing "
        "financial results.",
    ),
    _CorpusDocument(
        "cfo_prior_role.txt",
        "Before joining, Morgan Reyes worked at a supply chain analytics "
        "company as an operations executive.",
    ),
)

_RAW_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        "What was the total revenue in fiscal year 2024?",
        BenchmarkCategory.SIMPLE_FACTUAL,
        ("revenue_2024.txt",),
        ("150 million",),
    ),
    EvalCase(
        "What was the operating margin in fiscal year 2023?",
        BenchmarkCategory.SIMPLE_FACTUAL,
        ("revenue_2023.txt",),
        ("22 percent",),
    ),
    EvalCase(
        "How did revenue change between 2023 and 2024?",
        BenchmarkCategory.COMPARISON,
        ("revenue_2023.txt", "revenue_2024.txt"),
        ("increase",),
    ),
    EvalCase(
        "What changed in R&D spending from 2023 to 2024?",
        BenchmarkCategory.TEMPORAL,
        ("rd_spending.txt",),
        ("30 million", "25 million"),
    ),
    EvalCase(
        "Why might the company's revenue growth be at risk?",
        BenchmarkCategory.ANALYTICAL,
        ("risk_factors.txt",),
        ("competition", "risk"),
    ),
    EvalCase(
        "What was the combined revenue across fiscal years 2023 and 2024?",
        BenchmarkCategory.AGGREGATION,
        ("revenue_2023.txt", "revenue_2024.txt"),
        (),
    ),
    EvalCase("What about the numbers?", BenchmarkCategory.AMBIGUOUS, (), ()),
    EvalCase(
        "What is the CEO's favorite color?", BenchmarkCategory.UNANSWERABLE, (), ()
    ),
    EvalCase(
        "What was the Q3 2024 revenue decline percentage?",
        BenchmarkCategory.CONTRADICTORY_EVIDENCE,
        ("conflicting_revenue_a.txt", "conflicting_revenue_b.txt"),
        (),
    ),
    EvalCase(
        "Who was the CFO during fiscal year 2024, and what did they do before that?",
        BenchmarkCategory.MULTI_HOP,
        ("cfo_2024.txt", "cfo_prior_role.txt"),
        ("Morgan Reyes",),
    ),
)


async def build_benchmark_corpus(
    *,
    session: AsyncSession,
    object_store: ObjectStore,
    embedding_provider: EmbeddingProvider,
) -> tuple[uuid.UUID, list[BuiltEvalCase]]:
    """Ingests+indexes `_CORPUS_DOCUMENTS` into a fresh collection and
    resolves each case's `relevant_filenames` to the real `Document.id`
    values that ingestion just created."""
    collection = Collection(name=f"benchmark-{uuid.uuid4().hex[:8]}")
    session.add(collection)
    await session.flush()

    document_ids: dict[str, uuid.UUID] = {}
    chunking_config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL, chunk_size_tokens=150)
    for doc in _CORPUS_DOCUMENTS:
        result = await ingest_document(
            session=session,
            object_store=object_store,
            collection_id=collection.id,
            filename=doc.filename,
            content=doc.text.encode(),
            title=None,
            source=doc.source,
            max_upload_size_bytes=1_000_000,
        )
        await index_document_version(
            session=session,
            document=result.document,
            version=result.version,
            parsed=result.parsed,
            chunking_config=chunking_config,
            embedding_provider=embedding_provider,
        )
        document_ids[doc.filename] = result.document.id

    built_cases = [
        BuiltEvalCase(
            query=case.query,
            category=case.category,
            relevant_document_ids={document_ids[f] for f in case.relevant_filenames},
            expected_keywords=case.expected_keywords,
        )
        for case in _RAW_CASES
    ]
    return collection.id, built_cases
