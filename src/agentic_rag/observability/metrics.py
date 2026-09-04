"""Prometheus-compatible metrics (spec §31).

A thin, deliberately small set of instruments — one per thing spec §31
actually asks to track (latency per phase, tokens, retrieval iterations,
cache hits, failures) — rather than instrumenting everything reachable.
Exposed via `GET /metrics` in the standard Prometheus text exposition
format (`generate_latest()`), scrapeable by any Prometheus-compatible
collector without a custom exporter.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

QUERY_LATENCY_SECONDS = Histogram(
    "agentic_rag_query_latency_seconds", "Total /query pipeline latency", ["status"]
)
RETRIEVAL_LATENCY_SECONDS = Histogram(
    "agentic_rag_retrieval_latency_seconds", "Per-call retrieval latency"
)
RERANK_LATENCY_SECONDS = Histogram(
    "agentic_rag_rerank_latency_seconds", "Per-call reranking latency"
)
GENERATION_LATENCY_SECONDS = Histogram(
    "agentic_rag_generation_latency_seconds", "Synthesis + citation validation latency"
)
RETRIEVAL_ITERATIONS = Histogram(
    "agentic_rag_retrieval_iterations",
    "Agentic loop iterations per query",
    buckets=(1, 2, 3, 4, 5, 8),
)
ESTIMATED_TOKENS = Histogram("agentic_rag_estimated_tokens", "Estimated tokens per query")
CACHE_HITS = Counter("agentic_rag_cache_hits_total", "Cache hits", ["cache"])
CACHE_MISSES = Counter("agentic_rag_cache_misses_total", "Cache misses", ["cache"])
QUERY_FAILURES = Counter("agentic_rag_query_failures_total", "Query failures", ["reason"])
RERANK_FAILURES = Counter(
    "agentic_rag_rerank_failures_total",
    "Reranker errors (e.g. model load/inference failure) that fell back to "
    "unreranked, fusion-ordered results",
)
PROMPT_INJECTION_FLAGGED = Counter(
    "agentic_rag_prompt_injection_flagged_total",
    "Retrieved chunks excluded from a synthesis prompt due to a matched "
    "prompt-injection heuristic",
)


def render_metrics() -> tuple[bytes, str]:
    """Returns (body, content_type) ready to hand straight to an HTTP response."""
    return generate_latest(), CONTENT_TYPE_LATEST
