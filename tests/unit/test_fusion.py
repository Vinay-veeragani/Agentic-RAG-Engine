from agentic_rag.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_ranks_item_appearing_in_both_lists_highest() -> None:
    dense = ["a", "b", "c"]
    sparse = ["b", "a", "d"]

    fused = reciprocal_rank_fusion([dense, sparse])

    ids = [item for item, _ in fused]
    assert ids[0] in ("a", "b")  # both appear near the top of both lists
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_scores_agree_with_manual_formula() -> None:
    fused = dict(reciprocal_rank_fusion([["x", "y"]], k=60))
    assert fused["x"] == 1 / (60 + 1)
    assert fused["y"] == 1 / (60 + 2)


def test_rrf_sums_contributions_across_rankings() -> None:
    fused = dict(reciprocal_rank_fusion([["x"], ["x"]], k=60))
    assert fused["x"] == 2 * (1 / 61)


def test_rrf_item_only_in_one_ranking_still_included() -> None:
    fused = dict(reciprocal_rank_fusion([["x", "y"], ["z"]]))
    assert set(fused) == {"x", "y", "z"}


def test_rrf_empty_rankings_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_smaller_k_amplifies_rank_differences() -> None:
    fused_small_k = dict(reciprocal_rank_fusion([["x", "y"]], k=1))
    fused_large_k = dict(reciprocal_rank_fusion([["x", "y"]], k=1000))

    ratio_small = fused_small_k["x"] / fused_small_k["y"]
    ratio_large = fused_large_k["x"] / fused_large_k["y"]
    assert ratio_small > ratio_large > 1
