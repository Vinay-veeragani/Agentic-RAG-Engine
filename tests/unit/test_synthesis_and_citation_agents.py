import uuid

import pytest

from agentic_rag.agents.citation_agent import CitationAgent
from agentic_rag.agents.synthesis_agent import SynthesisAgent
from agentic_rag.agents.verifier import AnswerVerifier
from agentic_rag.core.models import AnswerStatus
from agentic_rag.generation.mock import MockLLMProvider
from agentic_rag.retrieval.base import RetrievedCandidate


def _candidate(content: str, **overrides) -> RetrievedCandidate:
    defaults = dict(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=content,
        page=None,
        section=None,
        heading=None,
        document_filename="doc.txt",
        document_title=None,
    )
    defaults.update(overrides)
    return RetrievedCandidate(**defaults)


@pytest.mark.asyncio
async def test_synthesis_agent_returns_insufficient_for_no_evidence() -> None:
    agent = SynthesisAgent(MockLLMProvider())
    result = await agent.synthesize("why did revenue decline", [])
    assert result.insufficient_evidence is True
    assert result.claims == []


@pytest.mark.asyncio
async def test_synthesis_agent_produces_claims_referencing_evidence() -> None:
    agent = SynthesisAgent(MockLLMProvider())
    evidence = [_candidate("Revenue declined due to weaker demand.")]
    result = await agent.synthesize("why did revenue decline", evidence)
    assert result.claims
    assert all(1 <= i <= len(evidence) for c in result.claims for i in c.evidence_indices)


@pytest.mark.asyncio
async def test_synthesis_agent_clamps_out_of_range_indices() -> None:
    from agentic_rag.agents.synthesis_agent import SynthesisResult, SynthesizedClaim

    class FakeLLM:
        model_name = "fake"

        async def complete_structured(self, **kwargs):
            return SynthesisResult(
                insufficient_evidence=False,
                claims=[SynthesizedClaim(text="claim", evidence_indices=[1, 5, 99])],
            )

    agent = SynthesisAgent(FakeLLM())
    evidence = [_candidate("only one chunk")]
    result = await agent.synthesize("q", evidence)
    assert result.claims[0].evidence_indices == [1]


@pytest.mark.asyncio
async def test_citation_agent_rejects_claim_with_no_citations() -> None:
    agent = CitationAgent(MockLLMProvider())
    result = await agent.validate_claim("some claim", [])
    assert result.entailed is False


@pytest.mark.asyncio
async def test_verifier_returns_grounded_answer_with_citations() -> None:
    llm = MockLLMProvider()
    verifier = AnswerVerifier(SynthesisAgent(llm), CitationAgent(llm))
    evidence = [_candidate("Revenue declined due to weaker enterprise demand.", page=5)]

    result = await verifier.generate("why did revenue decline", evidence)

    assert result.status == AnswerStatus.GROUNDED
    assert result.answer
    assert len(result.citations) >= 1
    assert result.citation_metrics is not None
    assert result.citation_metrics.claims_total >= 1


@pytest.mark.asyncio
async def test_verifier_returns_insufficient_evidence_for_no_evidence() -> None:
    llm = MockLLMProvider()
    verifier = AnswerVerifier(SynthesisAgent(llm), CitationAgent(llm))
    result = await verifier.generate("why did revenue decline", [])

    assert result.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.answer is None
    assert result.citations == []


@pytest.mark.asyncio
async def test_verifier_removes_unsupported_claims() -> None:
    from agentic_rag.agents.citation_agent import CitationValidation
    from agentic_rag.agents.synthesis_agent import SynthesisResult, SynthesizedClaim

    class FakeSynthesisLLM:
        model_name = "fake"

        async def complete_structured(self, **kwargs):
            return SynthesisResult(
                insufficient_evidence=False,
                claims=[
                    SynthesizedClaim(text="supported claim", evidence_indices=[1]),
                    SynthesizedClaim(text="unsupported claim", evidence_indices=[1]),
                ],
            )

    calls = {"n": 0}

    class SequencedCitationLLM:
        model_name = "fake"

        async def complete_structured(self, **kwargs):
            calls["n"] += 1
            entailed = calls["n"] == 1
            return CitationValidation(entailed=entailed, reason="stub")

    verifier = AnswerVerifier(
        SynthesisAgent(FakeSynthesisLLM()), CitationAgent(SequencedCitationLLM())
    )
    evidence = [_candidate("Revenue declined due to weaker demand.")]

    result = await verifier.generate("why did revenue decline", evidence)

    assert result.status == AnswerStatus.GROUNDED
    assert result.answer == "supported claim"
    assert result.removed_claims == ["unsupported claim"]
