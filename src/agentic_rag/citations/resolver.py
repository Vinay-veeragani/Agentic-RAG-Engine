"""Citation resolution — turns an LLM's claim-to-evidence-index
references into real, verifiable citations.

The LLM never sees or produces a real chunk_id/document_id: it only
references evidence by the small 1-based index it was shown in the prompt
(`[1]`, `[2]`, ...). This function is the *only* place those indices become
real database IDs, by looking them up in the same evidence list the prompt
was built from — so a citation can never point at a chunk that wasn't
actually retrieved: never fabricate citation references. Any index outside
the evidence list's range is simply dropped, never guessed at.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from agentic_rag.retrieval.base import RetrievedCandidate


@dataclass(slots=True)
class Citation:
    claim: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_filename: str
    page: int | None
    section: str | None
    source: str | None
    evidence_score: float | None
    url: str | None = None


def resolve_citations(
    claim_text: str, evidence_indices: list[int], evidence: list[RetrievedCandidate]
) -> list[Citation]:
    citations: list[Citation] = []
    for index in evidence_indices:
        if not (1 <= index <= len(evidence)):
            continue  # out-of-range reference — dropped, never fabricated
        candidate = evidence[index - 1]
        citations.append(
            Citation(
                claim=claim_text,
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                document_filename=candidate.document_filename,
                page=candidate.page,
                section=candidate.section,
                source=candidate.document_source,
                evidence_score=candidate.rerank_score
                if candidate.rerank_score is not None
                else candidate.fusion_score,
            )
        )
    return citations
