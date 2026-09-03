import uuid
from collections import Counter

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from agentic_rag.api.dependencies.db import DbSession
from agentic_rag.api.dependencies.embeddings import EmbeddingProviderDep
from agentic_rag.api.dependencies.storage import ObjectStoreDep
from agentic_rag.api.schemas.documents import (
    DocumentDetailResponse,
    DocumentIndexRequest,
    DocumentIndexResponse,
    DocumentIngestResponse,
    DocumentResponse,
    DocumentVersionResponse,
    ElementTypeCount,
)
from agentic_rag.chunking.base import ChunkingConfig
from agentic_rag.chunking.pipeline import index_document_version
from agentic_rag.core.config import get_settings
from agentic_rag.core.models import DocumentType
from agentic_rag.ingestion.parsers.base import get_parser
from agentic_rag.ingestion.pipeline import ingest_document
from agentic_rag.storage.models import Document, DocumentVersion

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


@router.post("/{document_id}/ingest", response_model=DocumentIndexResponse)
async def index_document(
    document_id: uuid.UUID,
    body: DocumentIndexRequest,
    db: DbSession,
    object_store: ObjectStoreDep,
    embedding_provider: EmbeddingProviderDep,
) -> DocumentIndexResponse:
    """Chunks + embeds + indexes a document version into pgvector — separate
    from POST /documents (which only parses + stores metadata), since this
    step is heavier and re-runnable with different chunking config."""
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")

    if body.version_id is not None:
        version = await db.get(DocumentVersion, body.version_id)
        if version is None or version.document_id != document.id:
            raise HTTPException(status_code=404, detail="document version not found")
    else:
        version = await db.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document.id)
            .order_by(DocumentVersion.version_number.desc())
        )
        if version is None:
            raise HTTPException(status_code=404, detail="document has no versions")

    raw_content = await object_store.load(version.storage_path)
    document_type = DocumentType(document.document_type)
    parsed = get_parser(document_type).parse(filename=document.filename, content=raw_content)

    config_overrides = body.model_dump(exclude={"version_id"}, exclude_none=True)
    chunking_config = ChunkingConfig(**config_overrides)

    chunks = await index_document_version(
        session=db,
        document=document,
        version=version,
        parsed=parsed,
        chunking_config=chunking_config,
        embedding_provider=embedding_provider,
    )
    await db.commit()

    parent_count = sum(1 for c in chunks if c.chunk_metadata.get("is_parent"))
    return DocumentIndexResponse(
        document_id=document.id,
        version_id=version.id,
        version_number=version.version_number,
        strategy=chunking_config.strategy,
        embedding_model=embedding_provider.model_name,
        embedding_dimensions=embedding_provider.dimensions,
        chunk_count=len(chunks),
        parent_chunk_count=parent_count,
    )
