"""SQLAlchemy ORM models for the full schema.

Design decision: all tables live in this one module rather than split across
knowledge/, citations/, evaluation/ etc. Domain modules import the ORM classes
they need from here and layer repository/business logic on top. This avoids
circular imports between e.g. `citations` and `knowledge` (a Citation FKs into
both DocumentChunk and Answer) and keeps the schema reviewable in one place.
Domain logic (parsers, chunkers, retrievers, agents) stays out of this file.

Embedding dimensionality is fixed per-deployment by `EMBEDDING_DIMENSIONS`
below, because pgvector columns have a static dimension. Swapping to an
embedding model with a different dimension requires a new migration that
alters `document_chunks.embedding` (and a full re-embed) — this is a real
constraint of pgvector, not something the provider abstraction can hide, and
is documented in docs/embeddings... (added when the embeddings phase lands).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    Computed,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentic_rag.storage.postgres import Base

# Default dimension for the local mock/sentence-transformers embedding provider
# (e.g. bge-small-en / all-MiniLM-L6-v2 family). See embeddings/base.py.
EMBEDDING_DIMENSIONS = 384


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(primary_key=True, default=uuid.uuid4)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(unique=True)
    display_name: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(unique=True)
    description: Mapped[str | None]
    retrieval_config: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    embedding_config: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    source_authority_config: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    documents: Mapped[list[Document]] = relationship(back_populates="collection")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE")
    )
    title: Mapped[str | None]
    source: Mapped[str | None]
    filename: Mapped[str]
    document_type: Mapped[str]
    language: Mapped[str | None]
    checksum: Mapped[str] = mapped_column(index=True)
    doc_metadata: Mapped[dict[str, object]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    ingested_at: Mapped[datetime | None]
    # The document's real-world date (publication/fiscal period), distinct
    # from created_at (upload time) — caller-supplied at upload time since
    # reliably extracting this from arbitrary prose is a real NLP problem,
    # not something a deterministic parser can do generically. MetadataFilter
    # .year (retrieval/filters.py) prefers this when set, falling back to
    # created_at's year only when it isn't — found missing entirely during
    # an engineering audit ("temporal" filtering was silently upload-date-only).
    document_date: Mapped[date | None]

    collection: Mapped[Collection] = relationship(back_populates="documents")
    versions: Mapped[list[DocumentVersion]] = relationship(back_populates="document")

    __table_args__ = (
        Index("ix_documents_collection_id", "collection_id"),
        CheckConstraint(
            "document_type IN ('pdf','docx','txt','markdown','html','csv','json')",
            name="ck_documents_document_type",
        ),
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    version_number: Mapped[int]
    checksum: Mapped[str]
    storage_path: Mapped[str]
    status: Mapped[str] = mapped_column(default="pending")
    chunking_config: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    embedding_config: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="versions")
    chunks: Mapped[list[DocumentChunk]] = relationship(back_populates="document_version")

    __table_args__ = (
        UniqueConstraint("document_id", "version_number", name="uq_document_version"),
        CheckConstraint(
            "status IN ('pending','parsed','indexed')",
            name="ck_document_versions_status",
        ),
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE")
    )
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="SET NULL")
    )
    chunk_index: Mapped[int]
    page: Mapped[int | None]
    section: Mapped[str | None]
    heading: Mapped[str | None]
    content: Mapped[str]
    token_count: Mapped[int]
    character_count: Mapped[int]
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    chunk_metadata: Mapped[dict[str, object]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Generated column backing sparse (full-text) retrieval — see retrieval/sparse.py.
    # "english" is a fixed text-search config for now; making it per-collection
    # configurable is future work, not needed yet.
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True), nullable=True
    )

    document_version: Mapped[DocumentVersion] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_document_chunks_document_version_id", "document_version_id"),
        Index("ix_document_chunks_document_id", "document_id"),
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_document_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
        # A version's chunk_index values are assigned by the chunker
        # (chunking/pipeline.py) and must be unique within that version — a
        # re-run bug producing a duplicate would otherwise silently double
        # up a chunk in retrieval with no schema-level guardrail.
        UniqueConstraint(
            "document_version_id", "chunk_index", name="uq_document_chunks_version_index"
        ),
    )


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = _uuid_pk()
    collection_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("collections.id"))
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    parent_query_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("queries.id"))
    trace_id: Mapped[str] = mapped_column(index=True)
    query_text: Mapped[str]
    query_type: Mapped[str | None]
    status: Mapped[str] = mapped_column(default="pending")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[datetime | None]


class QueryPlan(Base):
    __tablename__ = "query_plans"

    id: Mapped[uuid.UUID] = _uuid_pk()
    query_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("queries.id", ondelete="CASCADE"))
    strategy: Mapped[str]
    expand_query: Mapped[bool] = mapped_column(default=False)
    decompose: Mapped[bool] = mapped_column(default=False)
    max_iterations: Mapped[int]
    top_k: Mapped[int]
    filters: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    subqueries: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class RetrievalRun(Base):
    __tablename__ = "retrieval_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    query_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("queries.id", ondelete="CASCADE"))
    iteration_number: Mapped[int]
    retrieval_strategy: Mapped[str]
    query_text_used: Mapped[str]
    status: Mapped[str] = mapped_column(default="pending")
    candidate_count: Mapped[int] = mapped_column(default=0)
    error: Mapped[dict[str, object] | None] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[datetime | None]

    __table_args__ = (Index("ix_retrieval_runs_query_id", "query_id"),)


class RetrievedChunk(Base):
    __tablename__ = "retrieved_chunks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    retrieval_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("retrieval_runs.id", ondelete="CASCADE")
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="CASCADE")
    )
    dense_score: Mapped[float | None]
    sparse_score: Mapped[float | None]
    fusion_score: Mapped[float | None]
    rerank_score: Mapped[float | None]
    rank: Mapped[int]
    selected_as_evidence: Mapped[bool] = mapped_column(default=False)

    __table_args__ = (
        Index("ix_retrieved_chunks_retrieval_run_id", "retrieval_run_id"),
        # A retrieval run must record each chunk it surfaced at most once —
        # without this, a fusion/dedup bug could double-count a chunk's
        # contribution to a run with no schema-level guardrail.
        UniqueConstraint(
            "retrieval_run_id", "chunk_id", name="uq_retrieved_chunks_run_chunk"
        ),
    )


class EvidenceItem(Base):
    __tablename__ = "evidence_items"

    id: Mapped[uuid.UUID] = _uuid_pk()
    query_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("queries.id", ondelete="CASCADE"))
    retrieval_run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("retrieval_runs.id"))
    chunk_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("document_chunks.id"))
    relevance_score: Mapped[float | None]
    coverage_score: Mapped[float | None]
    sufficiency_judgement: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (Index("ix_evidence_items_query_id", "query_id"),)


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[uuid.UUID] = _uuid_pk()
    query_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("queries.id", ondelete="CASCADE"))
    answer_text: Mapped[str | None]
    status: Mapped[str]
    confidence: Mapped[float | None]
    token_usage: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    cost_estimate: Mapped[float | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    citations: Mapped[list[Citation]] = relationship(back_populates="answer")


class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    answer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("answers.id", ondelete="CASCADE"))
    claim_text: Mapped[str]
    chunk_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("document_chunks.id"))
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"))
    page: Mapped[int | None]
    section: Mapped[str | None]
    source: Mapped[str | None]
    url: Mapped[str | None]
    evidence_score: Mapped[float | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    answer: Mapped[Answer] = relationship(back_populates="citations")


class EvaluationDataset(Base):
    __tablename__ = "evaluation_datasets"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(unique=True)
    description: Mapped[str | None]
    category: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    cases: Mapped[list[EvaluationCase]] = relationship(back_populates="dataset")


class EvaluationCase(Base):
    __tablename__ = "evaluation_cases"

    id: Mapped[uuid.UUID] = _uuid_pk()
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="CASCADE")
    )
    query_text: Mapped[str]
    expected_answer: Mapped[str | None]
    expected_chunk_ids: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    category: Mapped[str | None]
    case_metadata: Mapped[dict[str, object]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    dataset: Mapped[EvaluationDataset] = relationship(back_populates="cases")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[uuid.UUID] = _uuid_pk()
    case_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluation_cases.id", ondelete="CASCADE")
    )
    pipeline: Mapped[str]
    metrics: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    raw_output: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (Index("ix_evaluation_results_case_id", "case_id"),)


class Event(Base):
    """Structured telemetry/streaming event log."""

    __tablename__ = "events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    query_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("queries.id", ondelete="CASCADE"))
    trace_id: Mapped[str] = mapped_column(index=True)
    event_type: Mapped[str]
    payload: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (Index("ix_events_query_id", "query_id"),)
