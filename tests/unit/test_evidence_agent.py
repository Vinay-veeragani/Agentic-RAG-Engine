import uuid

import pytest

from agentic_rag.agents.evidence_agent import EvidenceAgent
from agentic_rag.generation.mock import MockLLMProvider
from agentic_rag.retrieval.base import RetrievedCandidate


def _candidate(
    content: str, *, document_id=None, document_source=None, document_filename="doc.txt"
) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        content=content,
        page=None,
        section=None,
        heading=None,
        document_filename=document_filename,
        document_title=None,
        document_source=document_source,
    )


@pytest.mark.asyncio
async def test_evidence_agent_returns_insufficient_when_no_candidates() -> None:
    agent = EvidenceAgent(MockLLMProvider())
    result = await agent.evaluate("why did revenue decline", [])
    assert result.assessment.sufficient is False
    assert result.assessment.missing_information
    assert result.contradictions == []


@pytest.mark.asyncio
async def test_evidence_agent_sufficient_when_evidence_overlaps_query() -> None:
    agent = EvidenceAgent(MockLLMProvider())
    candidates = [_candidate("Revenue declined due to weaker enterprise demand.")]
    result = await agent.evaluate("why did revenue decline", candidates)
    assert result.assessment.sufficient is True
    assert result.assessment.missing_information == []


@pytest.mark.asyncio
async def test_evidence_agent_insufficient_when_evidence_unrelated() -> None:
    agent = EvidenceAgent(MockLLMProvider())
    candidates = [_candidate("Heavy rainfall is forecast across the region this week.")]
    result = await agent.evaluate("why did revenue decline", candidates)
    assert result.assessment.sufficient is False
    assert result.assessment.missing_information


@pytest.mark.asyncio
async def test_evidence_agent_scores_relevance_coverage_directness() -> None:
    agent = EvidenceAgent(MockLLMProvider())
    candidates = [_candidate("Revenue declined due to weaker enterprise demand.")]
    result = await agent.evaluate("why did revenue decline", candidates)
    assert 0.0 <= result.assessment.relevance <= 1.0
    assert 0.0 <= result.assessment.coverage <= 1.0
    assert 0.0 <= result.assessment.directness <= 1.0


@pytest.mark.asyncio
async def test_evidence_agent_detects_numeric_contradiction_across_documents() -> None:
    agent = EvidenceAgent(MockLLMProvider())
    candidates = [
        _candidate("Revenue declined 4% in Q3."),
        _candidate("Revenue declined 9% in Q3."),
    ]
    result = await agent.evaluate("what happened to revenue", candidates)
    assert len(result.contradictions) == 1
    assert result.contradictions[0].resolution is None  # no authority policy configured


@pytest.mark.asyncio
async def test_evidence_agent_does_not_flag_different_periods_as_contradiction() -> None:
    """Regression test: two documents reporting different margins for
    different fiscal years were being flagged as a contradiction — found by
    running the evaluation benchmark corpus, where a 2023 margin and a 2024
    margin legitimately differ (mixing periods without awareness is the bug,
    not the differing numbers themselves)."""
    agent = EvidenceAgent(MockLLMProvider())
    candidates = [
        _candidate("In fiscal year 2023, operating margin was 22 percent."),
        _candidate("In fiscal year 2024, operating margin was 25 percent."),
    ]
    result = await agent.evaluate("what was the operating margin", candidates)
    assert result.contradictions == []


@pytest.mark.asyncio
async def test_evidence_agent_detects_contradiction_with_spelled_out_percent() -> None:
    """Regression test: the metric pattern originally only matched a literal
    '%' symbol, missing real-world text (and this repo's own benchmark
    corpus) that spells out "percent" — found by actually running the
    evaluation benchmark against real corpus text, not a unit test written
    in isolation."""
    agent = EvidenceAgent(MockLLMProvider())
    candidates = [
        _candidate("Quarterly revenue declined 4 percent in Q3."),
        _candidate("Quarterly revenue declined 9 percent in Q3."),
    ]
    result = await agent.evaluate("what happened to revenue", candidates)
    assert len(result.contradictions) == 1


@pytest.mark.asyncio
async def test_evidence_agent_does_not_flag_same_document_as_contradiction() -> None:
    agent = EvidenceAgent(MockLLMProvider())
    shared_id = uuid.uuid4()
    candidates = [
        _candidate("Revenue declined 4% in Q3.", document_id=shared_id),
        _candidate("Revenue declined 4% in Q3, restated.", document_id=shared_id),
    ]
    result = await agent.evaluate("what happened to revenue", candidates)
    assert result.contradictions == []


@pytest.mark.asyncio
async def test_evidence_agent_resolves_contradiction_via_authority_order() -> None:
    agent = EvidenceAgent(
        MockLLMProvider(), source_authority_order=["annual report", "press release"]
    )
    candidates = [
        _candidate(
            "Revenue declined 4% in Q3.",
            document_source="Annual Report",
            document_filename="annual_report.pdf",
        ),
        _candidate(
            "Revenue declined 9% in Q3.",
            document_source="Press Release",
            document_filename="press_release.pdf",
        ),
    ]
    result = await agent.evaluate("what happened to revenue", candidates)
    assert len(result.contradictions) == 1
    resolution = result.contradictions[0].resolution
    assert resolution is not None
    assert "annual_report.pdf" in resolution


@pytest.mark.asyncio
async def test_evidence_agent_extracts_referenced_years() -> None:
    agent = EvidenceAgent(MockLLMProvider())
    candidates = [_candidate("Revenue in 2023 was strong; by 2025 it had declined.")]
    result = await agent.evaluate("how did revenue change", candidates)
    assert result.years_referenced == [2023, 2025]
    assert result.spans_multiple_periods is True
