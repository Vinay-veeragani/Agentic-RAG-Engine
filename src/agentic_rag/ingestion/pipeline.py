"""Ingestion orchestration: validate -> detect type -> parse -> persist.

This is the one place that turns "some bytes a user uploaded" into
`Document`/`DocumentVersion` rows plus a `ParsedDocument` for Phase 3
(chunking) to consume. It does not chunk or embed anything itself.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.ingestion.loaders.validation import detect_document_type, validate_upload
from agentic_rag.ingestion.metadata import compute_checksum
from agentic_rag.ingestion.parsed_document import ParsedDocument
from agentic_rag.ingestion.parsers.base import get_parser
from agentic_rag.storage.models import Document, DocumentVersion
from agentic_rag.storage.object_store import ObjectStore


@dataclass(slots=True)
class IngestResult:
    document: Document
    version: DocumentVersion
    parsed: ParsedDocument


async def ingest_document(
    *,
    session: AsyncSession,
    object_store: ObjectStore,
    collection_id: uuid.UUID,
    filename: str,
    content: bytes,
    title: str | None,
    source: str | None = None,
    max_upload_size_bytes: int,
) -> IngestResult:
    safe_name = validate_upload(filename, content, max_size_bytes=max_upload_size_bytes)
    document_type = detect_document_type(safe_name)
    checksum = compute_checksum(content)

    parsed = get_parser(document_type).parse(filename=safe_name, content=content)

    document = await session.scalar(
        select(Document).where(
            Document.collection_id == collection_id, Document.filename == safe_name
        )
    )
    if document is None:
        document = Document(
            collection_id=collection_id,
            title=title or parsed.title,
            source=source,
            filename=safe_name,
            document_type=document_type.value,
            checksum=checksum,
        )
        session.add(document)
        await session.flush()
        next_version_number = 1
    else:
        latest_version = await session.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document.id)
            .order_by(DocumentVersion.version_number.desc())
        )
        next_version_number = (latest_version.version_number + 1) if latest_version else 1
        document.checksum = checksum
        if title:
            document.title = title
        if source:
            document.source = source

    storage_key = f"{document.id}/v{next_version_number}/{safe_name}"
    storage_path = await object_store.save(storage_key, content)

    version = DocumentVersion(
        document_id=document.id,
        version_number=next_version_number,
        checksum=checksum,
        storage_path=storage_path,
        status="parsed",
        chunking_config={},
        embedding_config={},
    )
    session.add(version)
    await session.flush()

    return IngestResult(document=document, version=version, parsed=parsed)
