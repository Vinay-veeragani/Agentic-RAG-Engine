import uuid

from pydantic import BaseModel, Field

from agentic_rag.core.models import RetrievalStrategy
from agentic_rag.retrieval.base import MetadataFilter


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    filters: MetadataFilter = Field(default_factory=MetadataFilter)


class SearchResultItem(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_filename: str
    document_title: str | None
    page: int | None
    section: str | None
    heading: str | None
    content: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = Field(default=10, ge=1, le=50)
    candidate_pool_size: int = Field(default=30, ge=1, le=200)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    filters: MetadataFilter = Field(default_factory=MetadataFilter)


class RetrievedCandidateResponse(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_filename: str
    document_title: str | None
    page: int | None
    section: str | None
    heading: str | None
    content: str
    dense_score: float | None
    sparse_score: float | None
    fusion_score: float | None
    rank: int | None


class RetrieveResponse(BaseModel):
    query: str
    strategy: RetrievalStrategy
    results: list[RetrievedCandidateResponse]
