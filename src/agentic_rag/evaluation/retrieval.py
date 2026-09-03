"""Standard retrieval metrics (spec §33) — pure functions over a ranked list
of retrieved IDs and a set of relevant IDs. No DB, no LLM, no randomness:
same inputs always produce the same numbers, so these are testable with
hand-computed expected values.

By convention, an empty `relevant` set means every metric is 0.0 (there is
nothing to have recalled) — callers evaluating "did the system correctly
find nothing" (ambiguous/unanswerable benchmark cases) should check that
condition separately rather than reading these as ranking-quality scores.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def recall_at_k(retrieved: list[T], relevant: set[T], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[T], relevant: set[T], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = len(set(top_k) & relevant)
    return hits / len(top_k)


def mean_reciprocal_rank(retrieved: list[T], relevant: set[T]) -> float:
    if not relevant:
        return 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[T], relevant: set[T], k: int) -> float:
    """Binary relevance NDCG: every relevant item is equally "correct",
    there is no graded relevance score to weight by."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, item in enumerate(top_k, start=1) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(retrieved: list[T], relevant: set[T], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0
