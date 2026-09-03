import uuid

from agentic_rag.agents.retrieval_agent import _fuse_variants
from agentic_rag.retrieval.base import RetrievedCandidate


def _candidate(chunk_id, *, dense=None, sparse=None) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        content="text",
        page=None,
        section=None,
        heading=None,
        document_filename="doc.txt",
        document_title=None,
        dense_score=dense,
        sparse_score=sparse,
    )


def test_fuse_variants_ranks_chunk_appearing_in_multiple_variants_higher() -> None:
    shared_id = uuid.uuid4()
    only_in_first = uuid.uuid4()
    only_in_second = uuid.uuid4()

    variant_a = [_candidate(shared_id, dense=0.9), _candidate(only_in_first, dense=0.5)]
    variant_b = [_candidate(shared_id, sparse=0.8), _candidate(only_in_second, sparse=0.4)]

    fused = _fuse_variants([variant_a, variant_b])

    ids = [c.chunk_id for c in fused]
    assert ids[0] == shared_id
    assert set(ids) == {shared_id, only_in_first, only_in_second}


def test_fuse_variants_preserves_best_score_across_variants() -> None:
    chunk_id = uuid.uuid4()
    variant_a = [_candidate(chunk_id, dense=0.3)]
    variant_b = [_candidate(chunk_id, dense=0.9)]

    fused = _fuse_variants([variant_a, variant_b])

    assert fused[0].dense_score == 0.9


def test_fuse_variants_sets_fusion_score() -> None:
    chunk_id = uuid.uuid4()
    fused = _fuse_variants([[_candidate(chunk_id, dense=0.5)]])
    assert fused[0].fusion_score is not None
