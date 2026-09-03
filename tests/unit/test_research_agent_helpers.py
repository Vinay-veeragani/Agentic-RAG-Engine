import uuid

from agentic_rag.agents.research_agent import _dedupe_by_chunk_id, _refine_query
from agentic_rag.retrieval.base import RetrievedCandidate


def test_refine_query_folds_in_missing_information() -> None:
    refined = _refine_query("why did revenue decline", ["pricing pressure", "demand"])
    assert refined.startswith("why did revenue decline")
    assert "pricing pressure" in refined and "demand" in refined


def test_refine_query_falls_back_when_no_missing_information() -> None:
    refined = _refine_query("why did revenue decline", [])
    assert refined != "why did revenue decline"
    assert refined.startswith("why did revenue decline")


def _candidate(chunk_id) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        content="text",
        page=None,
        section=None,
        heading=None,
        document_filename="doc.txt",
        document_title=None,
    )


def test_dedupe_by_chunk_id_keeps_first_occurrence() -> None:
    shared_id = uuid.uuid4()
    first = _candidate(shared_id)
    duplicate = _candidate(shared_id)
    other = _candidate(uuid.uuid4())

    result = _dedupe_by_chunk_id([first, duplicate, other])

    assert len(result) == 2
    assert result[0] is first
