"""Citation metrics aggregated across a benchmark run — pure
arithmetic over the per-query `CitationQualityMetrics` already computed by
`citations/validator.py`; no new LLM calls here.

`citation_recall` needs a ground-truth "which chunks a correct
answer must cite" label per case, which this benchmark's fixtures don't
carry (they only label relevant *documents*, for retrieval metrics) — not
computed here; a documented gap, not silently approximated.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.citations.validator import CitationQualityMetrics


@dataclass(slots=True)
class AggregatedCitationMetrics:
    mean_precision: float
    mean_completeness: float
    cases_with_citations: int
    total_cases: int


def aggregate_citation_metrics(
    per_case: list[CitationQualityMetrics | None],
) -> AggregatedCitationMetrics:
    valid = [m for m in per_case if m is not None]
    if not valid:
        return AggregatedCitationMetrics(
            mean_precision=0.0,
            mean_completeness=0.0,
            cases_with_citations=0,
            total_cases=len(per_case),
        )
    return AggregatedCitationMetrics(
        mean_precision=sum(m.citation_precision for m in valid) / len(valid),
        mean_completeness=sum(m.citation_completeness for m in valid) / len(valid),
        cases_with_citations=len(valid),
        total_cases=len(per_case),
    )
