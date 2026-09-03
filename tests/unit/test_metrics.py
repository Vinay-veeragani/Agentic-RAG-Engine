from agentic_rag.observability.metrics import CACHE_HITS, QUERY_FAILURES, render_metrics


def test_render_metrics_returns_prometheus_text_format() -> None:
    body, content_type = render_metrics()
    assert b"python_gc_objects_collected_total" in body
    assert "text/plain" in content_type


def test_render_metrics_reflects_recorded_observations() -> None:
    CACHE_HITS.labels(cache="test_probe").inc()
    QUERY_FAILURES.labels(reason="TestError").inc()
    body, _ = render_metrics()
    assert b'cache="test_probe"' in body
    assert b'reason="TestError"' in body
