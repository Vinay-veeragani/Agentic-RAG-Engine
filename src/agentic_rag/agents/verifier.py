"""Groundedness verification — the final assembly step.

Reconstructs the answer from *only* the claims whose citations passed
entailment validation, rather than asking an LLM to "edit" its own prior
answer: since synthesis already produces discrete claims, dropping an
unsupported one and rejoining the rest is a deterministic operation, and one
fewer place a bad structured-output response could corrupt the final
answer. This is "if a claim cannot be supported: remove it" — implemented
literally, not as another prompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agentic_rag.agents.citation_agent import CitationAgent
from agentic_rag.agents.synthesis_agent import SynthesisAgent
from agentic_rag.citations.resolver import Citation, resolve_citations
from agentic_rag.citations.validator import CitationQualityMetrics, compute_citation_metrics
from agentic_rag.core.models import AnswerStatus
from agentic_rag.observability.events import EventEmitter, EventType
from agentic_rag.observability.metrics import GENERATION_LATENCY_SECONDS, PROMPT_INJECTION_FLAGGED
from agentic_rag.observability.tracing import get_logger
from agentic_rag.retrieval.base import RetrievedCandidate
from agentic_rag.security.prompt_injection import detect_injection_patterns

logger = get_logger(__name__)


def _filter_injected_evidence(
    evidence: list[RetrievedCandidate],
) -> list[RetrievedCandidate]:
    """Drops any candidate whose content matches a prompt-injection heuristic
    before it ever reaches a synthesis/citation prompt (see
    `security/prompt_injection.py`) — retrieved documents are untrusted data,
    so a chunk that looks like it's trying to instruct the model is excluded
    from evidence entirely rather than passed through with a warning label
    the model could still be influenced by."""
    clean: list[RetrievedCandidate] = []
    for candidate in evidence:
        matches = detect_injection_patterns(candidate.content)
        if matches:
            PROMPT_INJECTION_FLAGGED.inc()
            logger.warning(
                "evidence.prompt_injection_flagged",
                chunk_id=str(candidate.chunk_id),
                document_id=str(candidate.document_id),
                patterns=matches,
            )
        else:
            clean.append(candidate)
    return clean


@dataclass(slots=True)
class AnswerVerificationResult:
    status: AnswerStatus
    answer: str | None
    citations: list[Citation] = field(default_factory=list)
    removed_claims: list[str] = field(default_factory=list)
    citation_metrics: CitationQualityMetrics | None = None
    generation_latency_seconds: float = 0.0


class AnswerVerifier:
    def __init__(self, synthesis_agent: SynthesisAgent, citation_agent: CitationAgent) -> None:
        self._synthesis_agent = synthesis_agent
        self._citation_agent = citation_agent

    async def generate(
        self,
        query: str,
        evidence: list[RetrievedCandidate],
        *,
        emitter: EventEmitter | None = None,
    ) -> AnswerVerificationResult:
        start = time.perf_counter()
        evidence = _filter_injected_evidence(evidence)
        if emitter:
            emitter.emit(EventType.GENERATION_STARTED, evidence_count=len(evidence))
        result = await self._generate(query, evidence, emitter=emitter)
        result.generation_latency_seconds = time.perf_counter() - start
        GENERATION_LATENCY_SECONDS.observe(result.generation_latency_seconds)
        return result

    async def _generate(
        self,
        query: str,
        evidence: list[RetrievedCandidate],
        *,
        emitter: EventEmitter | None,
    ) -> AnswerVerificationResult:
        synthesis = await self._synthesis_agent.synthesize(query, evidence)

        if synthesis.insufficient_evidence or not synthesis.claims:
            return AnswerVerificationResult(status=AnswerStatus.INSUFFICIENT_EVIDENCE, answer=None)

        if emitter:
            emitter.emit(EventType.CITATION_VALIDATION_STARTED, claim_count=len(synthesis.claims))

        supported_claim_texts: list[str] = []
        removed_claims: list[str] = []
        all_citations: list[Citation] = []
        citations_total = 0
        citations_entailed = 0

        for claim in synthesis.claims:
            cited_evidence = [
                evidence[i - 1] for i in claim.evidence_indices if 1 <= i <= len(evidence)
            ]
            citation_count = len(cited_evidence)
            validation = await self._citation_agent.validate_claim(claim.text, cited_evidence)

            citations_total += citation_count
            if validation.entailed and cited_evidence:
                citations_entailed += citation_count
                supported_claim_texts.append(claim.text)
                all_citations.extend(
                    resolve_citations(claim.text, claim.evidence_indices, evidence)
                )
            else:
                removed_claims.append(claim.text)

        if not supported_claim_texts:
            return AnswerVerificationResult(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                answer=None,
                removed_claims=removed_claims,
            )

        metrics = compute_citation_metrics(
            claims_total=len(synthesis.claims),
            claims_supported=len(supported_claim_texts),
            citations_total=citations_total,
            citations_entailed=citations_entailed,
        )
        return AnswerVerificationResult(
            status=AnswerStatus.GROUNDED,
            answer=" ".join(supported_claim_texts),
            citations=all_citations,
            removed_claims=removed_claims,
            citation_metrics=metrics,
        )
