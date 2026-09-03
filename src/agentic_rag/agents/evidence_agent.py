"""Evidence sufficiency judgment (spec §15) — the lightweight version that
drives the Phase 7 agentic loop's continue/stop decision.

This deliberately covers only "does this evidence answer the query" for now.
The fuller judgment spec §15 describes — source quality/authority,
directness, temporal correctness, contradiction detection — is Phase 8's
scope (`EvidenceAssessment` will grow those fields then); building it here
would get ahead of the phase that's actually meant to implement it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_rag.generation.llm import LLMProvider
from agentic_rag.retrieval.base import RetrievedCandidate

_SYSTEM_PROMPT = (
    "You judge whether retrieved evidence sufficiently answers a query. "
    "Evidence that confirms WHAT happened but not WHY (when the query asks "
    "why) is NOT sufficient. Evidence that is only tangentially related is "
    "NOT sufficient. If insufficient, list the specific missing information "
    "needed, not a restatement of the query."
)


class EvidenceAssessment(BaseModel):
    sufficient: bool
    reason: str
    missing_information: list[str] = Field(default_factory=list)


class EvidenceAgent:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def assess(self, query: str, candidates: list[RetrievedCandidate]) -> EvidenceAssessment:
        if not candidates:
            return EvidenceAssessment(
                sufficient=False,
                reason="No evidence was retrieved for this query.",
                missing_information=[query],
            )

        evidence_text = "\n\n".join(
            f"[{i + 1}] {c.content}" for i, c in enumerate(candidates)
        )
        result = await self._llm.complete_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Query: {query}\n\nEvidence:\n{evidence_text}",
            schema=EvidenceAssessment,
        )
        return result
