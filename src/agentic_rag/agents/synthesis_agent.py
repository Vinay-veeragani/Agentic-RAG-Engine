"""Answer synthesis (spec §21).

The LLM is never asked to produce a real citation ID — only a small
1-based index into the evidence list it was shown in the prompt
(`evidence_indices: [1, 2]`). Turning those indices into real chunk/document
IDs happens entirely in `citations/resolver.py`, deterministically, outside
any LLM call — this is what makes "never fabricate a citation" an actual
guarantee rather than a prompt instruction hoping the model complies.

No evidence at all is a deterministic short-circuit (no LLM call, matching
`EvidenceAgent`'s and `MetadataFilter`'s established pattern) — there is
nothing to synthesize from.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_rag.generation.llm import LLMProvider
from agentic_rag.retrieval.base import RetrievedCandidate

_SYSTEM_PROMPT = (
    "You write a grounded answer to a query using ONLY the numbered evidence "
    "provided — never information you already know. Break your answer into "
    "discrete claims; for each claim, list the evidence numbers ([1], [2], "
    "...) that directly support it in `evidence_indices`. Never state a claim "
    "without at least one supporting evidence number. If the evidence does "
    "not actually answer the query, set insufficient_evidence to true and "
    "leave claims empty or limited to only what the evidence directly states."
)


class SynthesizedClaim(BaseModel):
    text: str
    evidence_indices: list[int] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    insufficient_evidence: bool
    claims: list[SynthesizedClaim] = Field(default_factory=list)


class SynthesisAgent:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def synthesize(
        self, query: str, evidence: list[RetrievedCandidate]
    ) -> SynthesisResult:
        if not evidence:
            return SynthesisResult(insufficient_evidence=True, claims=[])

        evidence_text = "\n\n".join(f"[{i + 1}] {c.content}" for i, c in enumerate(evidence))
        result = await self._llm.complete_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Query: {query}\n\nEvidence:\n{evidence_text}",
            schema=SynthesisResult,
        )
        # Never trust the model to have respected the evidence list's actual
        # bounds — indices are validated for real at citation-resolution time,
        # but claims with an obviously out-of-range-only reference set are
        # cheap to catch here too.
        max_index = len(evidence)
        for claim in result.claims:
            claim.evidence_indices = [i for i in claim.evidence_indices if 1 <= i <= max_index]
        return result
