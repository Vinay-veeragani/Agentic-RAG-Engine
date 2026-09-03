import uuid
from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID
    title: str | None
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
