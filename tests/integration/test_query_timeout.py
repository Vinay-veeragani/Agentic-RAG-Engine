"""`max_query_latency_seconds` is enforced with `asyncio.wait_for`
around the pipeline in `api/routes/query.py` — a hung provider must produce
a clean 504/TIMEOUT response, never a request that hangs forever."""

import asyncio
import uuid

import pytest

from agentic_rag.agents.research_agent import AgenticRetrievalLoop
from agentic_rag.core.config import Settings


@pytest.mark.asyncio
async def test_query_endpoint_returns_504_when_pipeline_exceeds_latency_budget(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny_timeout_settings = Settings(max_query_latency_seconds=0.05)
    monkeypatch.setattr(
        "agentic_rag.api.routes.query.get_settings", lambda: tiny_timeout_settings
    )

    async def hangs_forever(self, *args, **kwargs):
        await asyncio.sleep(5)
        raise AssertionError("should have been cancelled by the timeout")

    monkeypatch.setattr(AgenticRetrievalLoop, "run", hangs_forever)

    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]

    response = await client.post(
        "/query", json={"query": "why did revenue decline", "collection_id": collection_id}
    )

    assert response.status_code == 504
    assert response.json()["code"] == "TIMEOUT"
