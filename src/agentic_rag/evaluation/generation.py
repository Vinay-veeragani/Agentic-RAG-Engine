"""Generation-quality metrics (spec §33).

`faithfulness` and `context_relevance` are *not* re-judged here — they are
directly what `citations/validator.py`'s `citation_precision` (do the
claims that made it into the answer actually follow from their cited
evidence?) and Phase 8's `EvidenceAssessment.relevance` (was the retrieved
context relevant to the query?) already measure. Recomputing them with a
second LLM judge would be redundant, not more rigorous — the runner reuses
those existing values instead of adding a third pass here.

`answer_relevance` (does the final text actually address what was asked,
independent of whether it's grounded) is the one genuinely new judgment
this module adds, since nothing upstream measures it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_rag.generation.llm import LLMProvider

_SYSTEM_PROMPT = (
    "You judge whether an answer actually addresses what the query asked — "
    "independent of whether the answer is factually correct or grounded. "
    "An answer that is accurate but addresses a different question, or "
    "that hedges without engaging the question, is NOT relevant. Score "
    "from 0.0 (off-topic) to 1.0 (directly addresses the query)."
)


class AnswerRelevance(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)


class GenerationJudge:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def judge_answer_relevance(self, query: str, answer: str | None) -> float:
        if not answer or not answer.strip():
            return 0.0
        result = await self._llm.complete_structured(
            system_prompt=_SYSTEM_PROMPT,
            # Reuses the "Evidence:" label so MockLLMProvider's existing
            # term-overlap heuristic applies without a new prompt pattern —
            # the answer is, structurally, "the text being judged against
            # the query" here, the same role evidence plays elsewhere.
            user_prompt=f"Query: {query}\n\nEvidence:\n{answer}",
            schema=AnswerRelevance,
        )
        return result.relevance
