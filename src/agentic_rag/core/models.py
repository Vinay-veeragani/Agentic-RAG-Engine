"""Shared enums and small value types used across module boundaries.

These are the vocabulary the rest of the system is built on (query types,
retrieval strategies, failure modes). Keeping them here — rather than
duplicated per-module — is what lets api/schemas, storage models, and agent
outputs all agree on the same closed set of values.
"""

from __future__ import annotations

from enum import StrEnum


class FailureMode(StrEnum):
    """Closed set of ways a query can legitimately fail to produce a fabricated
    answer. Spec §35 — never silently convert one of these into a made-up answer."""

    NO_KNOWLEDGE = "NO_KNOWLEDGE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    TIMEOUT = "TIMEOUT"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    INVALID_DOCUMENT = "INVALID_DOCUMENT"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    PROMPT_INJECTION_DETECTED = "PROMPT_INJECTION_DETECTED"


class DocumentType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"
    JSON = "json"


class QueryType(StrEnum):
    """Query Analyzer output categories — spec §10."""

    SIMPLE_FACTUAL = "simple_factual"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    ANALYTICAL = "analytical"
    MULTI_HOP = "multi_hop"
    AMBIGUOUS = "ambiguous"
    UNANSWERABLE = "unanswerable"


class RetrievalStrategy(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class RetrievalRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class QueryStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
