"""Reciprocal Rank Fusion (spec §9) — a pure function over ranked ID lists,
with no DB/embedding/retriever dependency, so it's testable in complete
isolation from everything else in this package.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: list[list[T]], *, k: int = 60
) -> list[tuple[T, float]]:
    """Each inner list is one method's results, best-first. An item's fused
    score is the sum, across every ranking it appears in, of `1 / (k + rank)`
    (rank is 1-indexed). Returns `(item, score)` pairs sorted by score
    descending — items absent from a given ranking simply don't contribute a
    term for it, rather than being penalized with a worst-case rank.
    """
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
