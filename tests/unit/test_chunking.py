import pytest

from agentic_rag.chunking.base import ChunkingConfig, ChunkingStrategy, get_chunker
from agentic_rag.chunking.fixed import FixedSizeChunker
from agentic_rag.chunking.recursive import RecursiveChunker
from agentic_rag.chunking.structural import StructuralChunker
from agentic_rag.chunking.tokenization import count_tokens
from agentic_rag.embeddings.providers import MockEmbeddingProvider
from agentic_rag.ingestion.parsed_document import DocumentElement, ElementType, ParsedDocument


def _doc(elements: list[DocumentElement]) -> ParsedDocument:
    return ParsedDocument(filename="a.txt", document_type="txt", elements=elements)


def _paragraph(text: str, heading: str | None = None, index: int = 0) -> DocumentElement:
    return DocumentElement(
        element_type=ElementType.PARAGRAPH, text=text, order_index=index, heading=heading
    )


@pytest.mark.asyncio
async def test_fixed_chunker_respects_token_budget_and_overlap() -> None:
    long_text = " ".join(f"word{i}" for i in range(500))
    parsed = _doc([_paragraph(long_text)])
    config = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED, chunk_size_tokens=50, chunk_overlap_tokens=10
    )

    chunks = await FixedSizeChunker().chunk(parsed, config)

    assert len(chunks) > 1
    assert all(c.token_count <= 50 for c in chunks)
    # reconstructing without overlap should recover most of the original words
    assert "word0" in chunks[0].text
    assert "word499" in chunks[-1].text


@pytest.mark.asyncio
async def test_recursive_chunker_keeps_small_elements_together() -> None:
    parsed = _doc([_paragraph("Short one."), _paragraph("Short two."), _paragraph("Short three.")])
    config = ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE, chunk_size_tokens=100, chunk_overlap_tokens=0
    )

    chunks = await RecursiveChunker().chunk(parsed, config)

    assert len(chunks) == 1
    assert "Short one." in chunks[0].text and "Short three." in chunks[0].text


@pytest.mark.asyncio
async def test_recursive_chunker_splits_oversized_single_element() -> None:
    long_text = ". ".join(f"Sentence number {i}" for i in range(200))
    parsed = _doc([_paragraph(long_text)])
    config = ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE, chunk_size_tokens=50, chunk_overlap_tokens=0
    )

    chunks = await RecursiveChunker().chunk(parsed, config)

    assert len(chunks) > 1
    assert all(c.token_count <= 50 for c in chunks)


@pytest.mark.asyncio
async def test_structural_chunker_never_merges_across_headings() -> None:
    elements = [
        _paragraph("Intro content.", heading="Section A"),
        _paragraph("More intro.", heading="Section A"),
        _paragraph("Different topic.", heading="Section B"),
    ]
    parsed = _doc(elements)
    config = ChunkingConfig(
        strategy=ChunkingStrategy.STRUCTURAL, chunk_size_tokens=1000, chunk_overlap_tokens=0
    )

    chunks = await StructuralChunker().chunk(parsed, config)

    assert len(chunks) == 2
    assert chunks[0].heading == "Section A"
    assert chunks[1].heading == "Section B"
    assert "Different topic" not in chunks[0].text


@pytest.mark.asyncio
async def test_structural_chunker_creates_parent_and_child_for_oversized_section() -> None:
    long_text = ". ".join(f"Detail sentence {i}" for i in range(200))
    elements = [_paragraph(long_text, heading="Big Section")]
    parsed = _doc(elements)
    config = ChunkingConfig(
        strategy=ChunkingStrategy.STRUCTURAL, chunk_size_tokens=50, chunk_overlap_tokens=0
    )

    chunks = await StructuralChunker().chunk(parsed, config)

    parents = [c for c in chunks if c.metadata.get("is_parent")]
    children = [c for c in chunks if c.parent_index is not None]
    assert len(parents) == 1
    assert len(children) >= 2
    assert all(c.parent_index == 0 for c in children)
    assert parents[0].token_count > 50  # parent intentionally exceeds the budget


@pytest.mark.asyncio
async def test_semantic_chunker_breaks_on_low_similarity() -> None:
    from agentic_rag.chunking.semantic import SemanticChunker

    provider = MockEmbeddingProvider()
    elements = [_paragraph("First sentence. Second sentence. Third sentence.")]
    parsed = _doc(elements)
    config = ChunkingConfig(
        strategy=ChunkingStrategy.SEMANTIC,
        chunk_size_tokens=1000,
        semantic_similarity_threshold=1.0,  # effectively unsatisfiable for distinct mock vectors
    )

    chunks = await SemanticChunker(provider).chunk(parsed, config)
    assert len(chunks) == 3


def test_get_chunker_returns_expected_types() -> None:
    assert isinstance(get_chunker(ChunkingStrategy.FIXED), FixedSizeChunker)
    assert isinstance(get_chunker(ChunkingStrategy.RECURSIVE), RecursiveChunker)
    assert isinstance(get_chunker(ChunkingStrategy.STRUCTURAL), StructuralChunker)


def test_get_chunker_semantic_requires_embedding_provider() -> None:
    with pytest.raises(ValueError):
        get_chunker(ChunkingStrategy.SEMANTIC)


def test_count_tokens_is_deterministic() -> None:
    assert count_tokens("hello world") == count_tokens("hello world")
    assert count_tokens("") == 0
