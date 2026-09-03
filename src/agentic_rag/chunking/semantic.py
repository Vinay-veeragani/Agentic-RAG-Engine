from __future__ import annotations

import re

import numpy as np

from agentic_rag.chunking.base import ChunkCandidate, ChunkingConfig
from agentic_rag.chunking.tokenization import count_tokens
from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.ingestion.parsed_document import DocumentElement, ParsedDocument

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker:
    """Splits each element into sentences, embeds them, and merges adjacent
    sentences into a chunk until either the token budget is hit or the
    cosine similarity to the next sentence drops below
    `config.semantic_similarity_threshold` — a topic-shift breakpoint,
    rather than a fixed size or a structural boundary.

    Requires an `EmbeddingProvider`, unlike the other three chunkers — this
    is the one strategy where "reasoning" (semantic similarity) genuinely
    does something a purely deterministic split can't (spec principle #2).
    """

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embeddings = embedding_provider

    async def chunk(self, parsed: ParsedDocument, config: ChunkingConfig) -> list[ChunkCandidate]:
        elements = parsed.text_elements
        if not elements:
            return []

        sentences, owners = _split_into_sentences(elements)
        if not sentences:
            return []
        if len(sentences) == 1:
            return _finalize([sentences[0]], [owners[0]], 0)

        vectors = np.array(await self._embeddings.embed_texts(sentences))
        similarities = _consecutive_cosine_similarities(vectors)

        candidates: list[ChunkCandidate] = []
        group_sentences: list[str] = [sentences[0]]
        group_owners: list[DocumentElement] = [owners[0]]

        for i in range(1, len(sentences)):
            candidate_text = " ".join([*group_sentences, sentences[i]])
            similarity = similarities[i - 1]
            fits_budget = count_tokens(candidate_text) <= config.chunk_size_tokens
            same_topic = similarity >= config.semantic_similarity_threshold

            if fits_budget and same_topic:
                group_sentences.append(sentences[i])
                group_owners.append(owners[i])
            else:
                candidates.extend(_finalize(group_sentences, group_owners, len(candidates)))
                group_sentences = [sentences[i]]
                group_owners = [owners[i]]

        candidates.extend(_finalize(group_sentences, group_owners, len(candidates)))
        return candidates


def _split_into_sentences(
    elements: list[DocumentElement],
) -> tuple[list[str], list[DocumentElement]]:
    sentences: list[str] = []
    owners: list[DocumentElement] = []
    for element in elements:
        for sentence in _SENTENCE_SPLIT.split(element.text.strip()):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
                owners.append(element)
    return sentences, owners


def _consecutive_cosine_similarities(vectors: np.ndarray) -> list[float]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    return [float(np.dot(unit[i], unit[i + 1])) for i in range(len(unit) - 1)]


def _finalize(
    sentences: list[str], owners: list[DocumentElement], order_index: int
) -> list[ChunkCandidate]:
    if not sentences:
        return []
    text = " ".join(sentences)
    anchor = owners[0]
    return [
        ChunkCandidate(
            text=text,
            order_index=order_index,
            token_count=count_tokens(text),
            character_count=len(text),
            page=anchor.page,
            section=anchor.section,
            heading=anchor.heading,
        )
    ]
