import uuid

import pytest

from agentic_rag.agents.evidence_agent import EvidenceAgent
from agentic_rag.generation.mock import MockLLMProvider
from agentic_rag.retrieval.base import RetrievedCandidate


def _candidate(content: str) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=content,
        page=None,
        section=None,
        heading=None,
        document_filename="doc.txt",
        document_title=None,
    )


@pytest.mark.asyncio
async def test_evidence_agent_returns_insufficient_when_no_candidates() -> None:
    agent = EvidenceAgent(MockLLMProvider())
    result = await agent.assess("why did revenue decline", [])
    assert result.sufficient is False
    assert result.missing_information


@pytest.mark.asyncio
async def test_evidence_agent_sufficient_when_evidence_overlaps_query() -> None:
    agent = EvidenceAgent(MockLLMProvider())
    candidates = [_candidate("Revenue declined due to weaker enterprise demand.")]
    result = await agent.assess("why did revenue decline", candidates)
    assert result.sufficient is True
    assert result.missing_information == []


@pytest.mark.asyncio
async def test_evidence_agent_insufficient_when_evidence_unrelated() -> None:
    agent = EvidenceAgent(MockLLMProvider())
    candidates = [_candidate("Heavy rainfall is forecast across the region this week.")]
    result = await agent.assess("why did revenue decline", candidates)
    assert result.sufficient is False
    assert result.missing_information
