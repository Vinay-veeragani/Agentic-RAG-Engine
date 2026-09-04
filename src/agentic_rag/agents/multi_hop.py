"""Dependency-chained multi-hop retrieval.

Plain query decomposition (`agents/planner.py::QueryDecomposer`) splits a
query into subqueries and retrieves each one *independently*, then fuses the
rankings — real and useful for a query like "what are Acme's revenue and
profit", where both halves can be searched for directly. It cannot answer
the harder multi-hop shape ("find companies -> then look
up their values"): a second hop that depends on an entity the *first* hop's
evidence reveals, which the original query never names at all (e.g. "what
year was Acme Corp's CEO born" — no document mentions "Acme Corp's CEO" by
name; it mentions "Jordan Smith", and only the first hop's evidence reveals
that). An independent-subquery search for hop two's literal text can't find
that evidence; nothing links the two searches.

This module closes that gap with exactly two hops (a genuinely larger
dependency chain is future work, not attempted here): hop one runs a normal
retrieval, an LLM call extracts the single bridging entity hop two actually
needs from hop one's evidence, and hop two's query is built by deterministic
string substitution (extraction needs real language understanding; slotting
the extracted text into the next query does not, per this project's
deterministic-first principle) before running a second, real retrieval call
against that resolved query.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentic_rag.generation.llm import LLMProvider
from agentic_rag.retrieval.base import RetrievedCandidate

_SYSTEM_PROMPT = (
    "A multi-hop question is being answered in two steps. You have the "
    "evidence retrieved for the first step. Extract the single key entity, "
    "name, or value from that evidence that the second step needs in order "
    "to search for its answer — e.g. a person's name, a company, an "
    "identifier. Return just that entity, nothing else. If no evidence is "
    "given or no clear entity is present, return an empty string."
)


class HopEntity(BaseModel):
    entity: str


class MultiHopResolver:
    """Wraps one LLM call: given the first hop's query and evidence, extract
    the entity the second hop's query should be resolved against."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def extract_bridge_entity(
        self, first_hop_query: str, first_hop_evidence: list[RetrievedCandidate]
    ) -> str:
        if not first_hop_evidence:
            return ""
        evidence_text = "\n\n".join(
            f"[{i + 1}] {c.content}" for i, c in enumerate(first_hop_evidence)
        )
        result = await self._llm.complete_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Query: {first_hop_query}\n\nEvidence:\n{evidence_text}",
            schema=HopEntity,
        )
        return result.entity.strip()


def resolve_second_hop_query(second_hop_query: str, entity: str) -> str:
    """Deterministic: fold the extracted entity into the second hop's query
    text. A bare append is a defensible, simple strategy (same tradeoff as
    `research_agent.py::_refine_query`'s query refinement) — no LLM call
    needed to combine two strings."""
    entity = entity.strip()
    if not entity:
        return second_hop_query
    return f"{second_hop_query} {entity}"
