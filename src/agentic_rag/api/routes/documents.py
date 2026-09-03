import uuid
from collections import Counter

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from agentic_rag.api.dependencies.db import DbSession
from agentic_rag.api.dependencies.storage import ObjectStoreDep
from agentic_rag.api.schemas.documents import (
    DocumentDetailResponse,
    DocumentIngestResponse,
    DocumentResponse,
    DocumentVersionResponse,
    ElementTypeCount,
)
from agentic_rag.core.config import get_settings
from agentic_rag.ingestion.pipeline import ingest_document
from agentic_rag.storage.models import Document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentIngestResponse, status_code=201)
async def upload_document(
    db: DbSession,
    object_store: ObjectStoreDep,
    collection_id: uuid.UUID = Form(...),
    file: UploadFile = File(...),
    title: str | None = Form(None),
) -> DocumentIngestResponse:
    content = await file.read()
    result = await ingest_document(
        session=db,
        object_store=object_store,
        collection_id=collection_id,
        filename=file.filename or "upload",
        content=content,
        title=title,
        max_upload_size_bytes=get_settings().max_upload_size_bytes,
    )
    await db.commit()
    await db.refresh(result.document)
    await db.refresh(result.version)

    type_counts = Counter(e.element_type.value for e in result.parsed.elements)
    return DocumentIngestResponse(
        document=DocumentResponse.model_validate(result.document),
        version=DocumentVersionResponse.model_validate(result.version),
        element_count=len(result.parsed.elements),
        page_count=result.parsed.page_count,
        element_type_counts=[
            ElementTypeCount(element_type=t, count=c) for t, c in type_counts.items()
        ],
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(db: DbSession, collection_id: uuid.UUID | None = None) -> list[Document]:
    query = select(Document).order_by(Document.created_at.desc())
    if collection_id is not None:
        query = query.where(Document.collection_id == collection_id)
    result = await db.scalars(query)
    return list(result.all())


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(document_id: uuid.UUID, db: DbSession) -> Document:
    document = await db.scalar(
        select(Document)
        .options(selectinload(Document.versions))
        .where(Document.id == document_id)
    )
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return document
