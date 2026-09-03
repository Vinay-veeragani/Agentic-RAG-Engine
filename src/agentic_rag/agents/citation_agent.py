"""Citation validation (spec §23) — does a claim's cited evidence actually
entail it, not merely relate to it?

A claim with zero citations is trivially unsupported (deterministic
short-circuit, no LLM call needed — there is nothing to check entailment
against). Otherwise this is a genuine language-understanding judgment
(entailment), so it goes through the LLM, same as evidence sufficiency.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentic_rag.generation.llm import LLMProvider
from agentic_rag.retrieval.base import RetrievedCandidate

_SYSTEM_PROMPT = (
    "You check whether the cited evidence actually entails (directly "
    "supports) the claim — not merely relates to the same topic. A claim "
    "that goes beyond what the evidence states, or that the evidence only "
    "tangentially touches on, is NOT entailed."
)


class CitationValidation(BaseModel):
    entailed: bool
    reason: str


class CitationAgent:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def validate_claim(
        self, claim_text: str, cited_evidence: list[RetrievedCandidate]
    ) -> CitationValidation:
        if not cited_evidence:
            return CitationValidation(
                entailed=False, reason="Claim has no supporting citation."
            )

        evidence_text = "\n\n".join(c.content for c in cited_evidence)
        return await self._llm.complete_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Claim: {claim_text}\n\nEvidence:\n{evidence_text}",
            schema=CitationValidation,
        )
