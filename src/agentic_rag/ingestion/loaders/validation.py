"""Upload validation: file type detection, size limits, filename safety
(spec §36 — file type validation, file size limits, path traversal protection).

This is deliberately deterministic (extension/size checks), not an LLM call —
per engineering principle #1, prefer deterministic logic where it's sufficient.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from agentic_rag.core.errors import InvalidDocumentError, UnsupportedFileTypeError
from agentic_rag.core.models import DocumentType

_EXTENSION_TO_TYPE: dict[str, DocumentType] = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
    ".txt": DocumentType.TXT,
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
    ".html": DocumentType.HTML,
    ".htm": DocumentType.HTML,
    ".csv": DocumentType.CSV,
    ".json": DocumentType.JSON,
}


def sanitize_filename(filename: str) -> str:
    """Strips any directory components — never trust a client-supplied path.

    Rejects the empty/root result rather than silently substituting a name,
    since that would let two different uploads collide under one filename.
    """
    name = PurePosixPath(filename.replace("\\", "/")).name
    if not name or name in (".", ".."):
        raise InvalidDocumentError(f"invalid filename: {filename!r}")
    return name


def detect_document_type(filename: str) -> DocumentType:
    suffix = PurePosixPath(filename).suffix.lower()
    document_type = _EXTENSION_TO_TYPE.get(suffix)
    if document_type is None:
        raise UnsupportedFileTypeError(
            f"unsupported file extension: {suffix!r}",
            details={"filename": filename},
        )
    return document_type


def validate_upload(filename: str, content: bytes, *, max_size_bytes: int) -> str:
    """Runs all upload-time checks; returns the sanitized filename.

    Raises `InvalidDocumentError` for empty/oversized content and
    `UnsupportedFileTypeError` for an unrecognized extension.
    """
    safe_name = sanitize_filename(filename)
    detect_document_type(safe_name)  # raises UnsupportedFileTypeError if unknown

    if len(content) == 0:
        raise InvalidDocumentError("uploaded file is empty", details={"filename": safe_name})
    if len(content) > max_size_bytes:
        raise InvalidDocumentError(
            f"uploaded file exceeds max size of {max_size_bytes} bytes",
            details={"filename": safe_name, "size": len(content)},
        )
    return safe_name
