import uuid
from datetime import datetime

from pydantic import BaseModel

from agentic_rag.chunking.base import ChunkingStrategy


class DocumentResponse(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID
    title: str | None
    source: str | None
    filename: str
    document_type: str
    checksum: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentVersionResponse(BaseModel):
    id: uuid.UUID
    version_number: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentDetailResponse(DocumentResponse):
    versions: list[DocumentVersionResponse]


class ElementTypeCount(BaseModel):
    element_type: str
    count: int


class DocumentIngestResponse(BaseModel):
    document: DocumentResponse
    version: DocumentVersionResponse
    element_count: int
    page_count: int | None
    element_type_counts: list[ElementTypeCount]


class DocumentIndexRequest(BaseModel):
    """All fields optional — omitted ones fall back to config defaults, so a
    plain `{}` body indexes with the platform's default chunking config."""

    version_id: uuid.UUID | None = None
    strategy: ChunkingStrategy | None = None
    chunk_size_tokens: int | None = None
    chunk_overlap_tokens: int | None = None
    min_chunk_size_tokens: int | None = None
    semantic_similarity_threshold: float | None = None


class DocumentIndexResponse(BaseModel):
    document_id: uuid.UUID
    version_id: uuid.UUID
    version_number: int
    strategy: ChunkingStrategy
    embedding_model: str
    embedding_dimensions: int
    chunk_count: int
    parent_chunk_count: int
