import uuid

from agentic_rag.core.models import DocumentType
from agentic_rag.retrieval.base import MetadataFilter
from agentic_rag.retrieval.filters import build_filter_conditions


def test_no_filter_produces_no_conditions() -> None:
    assert build_filter_conditions(None) == []
    assert build_filter_conditions(MetadataFilter()) == []


def test_each_field_contributes_exactly_one_condition() -> None:
    filters = MetadataFilter(
        collection_id=uuid.uuid4(),
        document_type=DocumentType.PDF,
        document_ids=[uuid.uuid4()],
        section="Intro",
        heading="Overview",
        source="10-K",
        year=2025,
    )
    conditions = build_filter_conditions(filters)
    assert len(conditions) == 7


def test_partial_filter_only_produces_conditions_for_set_fields() -> None:
    filters = MetadataFilter(section="Risks")
    conditions = build_filter_conditions(filters)
    assert len(conditions) == 1
