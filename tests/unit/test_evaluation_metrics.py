from agentic_rag.citations.validator import compute_citation_metrics
from agentic_rag.evaluation.citations import aggregate_citation_metrics
from agentic_rag.evaluation.retrieval import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_hand_computed() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"c", "z"}  # z never retrieved
    assert recall_at_k(retrieved, relevant, k=5) == 0.5
    assert recall_at_k(retrieved, relevant, k=2) == 0.0


def test_precision_at_k_hand_computed() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c", "e"}
    assert precision_at_k(retrieved, relevant, k=5) == 3 / 5
    assert precision_at_k(retrieved, relevant, k=1) == 1.0


def test_precision_at_k_empty_retrieved_list() -> None:
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_mrr_first_relevant_position() -> None:
    assert mean_reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert mean_reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_perfect_ranking_is_one() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert ndcg_at_k(retrieved, relevant, k=3) == 1.0


def test_ndcg_worse_ranking_scores_lower_than_perfect() -> None:
    perfect = ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3)
    worse = ndcg_at_k(["c", "a", "b"], {"a", "b"}, k=3)
    assert worse < perfect


def test_hit_rate_at_k() -> None:
    assert hit_rate_at_k(["a", "b"], {"b"}, k=2) == 1.0
    assert hit_rate_at_k(["a", "b"], {"z"}, k=2) == 0.0


def test_all_metrics_return_zero_for_empty_relevant_set() -> None:
    retrieved = ["a", "b", "c"]
    assert recall_at_k(retrieved, set(), k=3) == 0.0
    assert precision_at_k(retrieved, set(), k=3) == 0.0
    assert mean_reciprocal_rank(retrieved, set()) == 0.0
    assert ndcg_at_k(retrieved, set(), k=3) == 0.0
    assert hit_rate_at_k(retrieved, set(), k=3) == 0.0


def test_aggregate_citation_metrics_averages_across_cases() -> None:
    m1 = compute_citation_metrics(
        claims_total=2, claims_supported=2, citations_total=2, citations_entailed=2
    )
    m2 = compute_citation_metrics(
        claims_total=2, claims_supported=1, citations_total=2, citations_entailed=1
    )
    aggregated = aggregate_citation_metrics([m1, m2, None])
    assert aggregated.total_cases == 3
    assert aggregated.cases_with_citations == 2
    assert aggregated.mean_precision == 0.75  # (1.0 + 0.5) / 2
    assert aggregated.mean_completeness == 0.75  # (1.0 + 0.5) / 2


def test_aggregate_citation_metrics_handles_all_none() -> None:
    aggregated = aggregate_citation_metrics([None, None])
    assert aggregated.mean_precision == 0.0
    assert aggregated.mean_completeness == 0.0
    assert aggregated.cases_with_citations == 0
    assert aggregated.total_cases == 2
