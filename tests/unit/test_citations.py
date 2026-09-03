import uuid

from agentic_rag.citations.formatter import format_citation
from agentic_rag.citations.resolver import resolve_citations
from agentic_rag.citations.validator import compute_citation_metrics
from agentic_rag.retrieval.base import RetrievedCandidate


def _candidate(**overrides) -> RetrievedCandidate:
    defaults = dict(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Some content.",
        page=42,
        section="Revenue Recognition",
        heading=None,
        document_filename="annual_report.pdf",
        document_title=None,
        document_source="Annual Report",
        rerank_score=0.8,
    )
    defaults.update(overrides)
    return RetrievedCandidate(**defaults)


def test_resolve_citations_maps_valid_indices() -> None:
    evidence = [_candidate(), _candidate()]
    citations = resolve_citations("some claim", [1, 2], evidence)
    assert len(citations) == 2
    assert citations[0].chunk_id == evidence[0].chunk_id
    assert citations[1].chunk_id == evidence[1].chunk_id


def test_resolve_citations_drops_out_of_range_indices() -> None:
    evidence = [_candidate()]
    citations = resolve_citations("some claim", [1, 2, 99, 0, -1], evidence)
    assert len(citations) == 1
    assert citations[0].chunk_id == evidence[0].chunk_id


def test_resolve_citations_empty_indices_returns_empty() -> None:
    assert resolve_citations("some claim", [], [_candidate()]) == []


def test_resolve_citations_never_invents_ids_for_empty_evidence() -> None:
    assert resolve_citations("some claim", [1, 2, 3], []) == []


def test_format_citation_includes_source_page_and_section() -> None:
    evidence = [_candidate()]
    citations = resolve_citations("claim", [1], evidence)
    label = format_citation(citations[0], 1)
    assert label == "[1] Annual Report, page 42, Revenue Recognition"


def test_format_citation_falls_back_to_filename_without_source() -> None:
    evidence = [_candidate(document_source=None, page=None, section=None)]
    citations = resolve_citations("claim", [1], evidence)
    label = format_citation(citations[0], 1)
    assert label == "[1] annual_report.pdf"


def test_citation_metrics_completeness_and_precision() -> None:
    metrics = compute_citation_metrics(
        claims_total=4, claims_supported=3, citations_total=5, citations_entailed=4
    )
    assert metrics.citation_completeness == 0.75
    assert metrics.citation_precision == 0.8


def test_citation_metrics_handles_zero_claims() -> None:
    metrics = compute_citation_metrics(
        claims_total=0, claims_supported=0, citations_total=0, citations_entailed=0
    )
    assert metrics.citation_completeness == 0.0
    assert metrics.citation_precision == 0.0
