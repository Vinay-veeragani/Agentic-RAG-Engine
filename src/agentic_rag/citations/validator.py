"""Citation quality metrics (spec §23) — pure arithmetic over already-computed
validation results, no LLM call here.

`citation_recall` (spec §33) needs a ground-truth relevant-chunk set from an
evaluation dataset and isn't computable from a single live query — that
metric belongs to the Phase 10 evaluation framework, not here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CitationQualityMetrics:
    claims_total: int
    claims_supported: int  # claims with >=1 citation that passed entailment validation
    citations_total: int
    citations_entailed: int

    @property
    def citation_completeness(self) -> float:
        """Fraction of claims that ended up with at least one validated citation."""
        return self.claims_supported / self.claims_total if self.claims_total else 0.0

    @property
    def citation_precision(self) -> float:
        """Fraction of proposed citations that were actually entailed by their claim."""
        return self.citations_entailed / self.citations_total if self.citations_total else 0.0


def compute_citation_metrics(
    *, claims_total: int, claims_supported: int, citations_total: int, citations_entailed: int
) -> CitationQualityMetrics:
    return CitationQualityMetrics(
        claims_total=claims_total,
        claims_supported=claims_supported,
        citations_total=citations_total,
        citations_entailed=citations_entailed,
    )
