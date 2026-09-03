"""The baseline pipeline the evaluation framework compares the agentic
system against (spec §34): Query -> Dense Retrieval -> Top-K -> LLM. No
query analysis, no planning, no reranking, no evidence judgment, no
citation validation, no bounded refinement loop — deliberately the
simplest thing that could be called "RAG," so the comparison actually
means something.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.embeddings.base import EmbeddingProvider
from agentic_rag.generation.llm import LLMProvider
from agentic_rag.retrieval.base import MetadataFilter
from agentic_rag.retrieval.dense import DenseRetriever

_SYSTEM_PROMPT = "Answer the query using only the provided context."

DEFAULT_TOP_K = 5


@dataclass(slots=True)
class BaselineResult:
    answer: str | None
    document_ids: list[uuid.UUID]
    latency_seconds: float


class BaselinePipeline:
    def __init__(
        self, session: AsyncSession, embedding_provider: EmbeddingProvider, llm: LLMProvider
    ) -> None:
        self._session = session
        self._embeddings = embedding_provider
        self._llm = llm

    async def run(
        self, query: str, *, collection_id: uuid.UUID | None, top_k: int = DEFAULT_TOP_K
    ) -> BaselineResult:
        start = time.perf_counter()
        retriever = DenseRetriever(self._session, self._embeddings)
        candidates = await retriever.retrieve(
            query, top_k=top_k, filters=MetadataFilter(collection_id=collection_id)
        )

        if not candidates:
            return BaselineResult(
                answer=None, document_ids=[], latency_seconds=time.perf_counter() - start
            )

        context = "\n\n".join(c.content for c in candidates)
        answer = await self._llm.complete(
            system_prompt=_SYSTEM_PROMPT, user_prompt=f"Query: {query}\n\nEvidence:\n{context}"
        )
        return BaselineResult(
            answer=answer,
            document_ids=[c.document_id for c in candidates],
            latency_seconds=time.perf_counter() - start,
        )
