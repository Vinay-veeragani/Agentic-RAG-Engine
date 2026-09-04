"""Upload validation: file type detection (extension + magic-byte content
check for binary formats), size limits, filename safety.

This is deliberately deterministic (extension/size/signature checks), not
an LLM call — per engineering principle #1, prefer deterministic logic
where it's sufficient.
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

# Magic-byte signatures for the formats that have one — deliberately not
# using a dependency like python-magic (per engineering principle #1: a
# stdlib check is sufficient here). Only binary formats are checked: a
# text format (TXT/MD/HTML/CSV/JSON) has no reliable universal signature,
# and rejecting on content sniffing there would risk false positives on
# legitimate files with unusual encodings — the actual parser already
# fails loudly on genuinely unparseable content. This catches the real,
# cheap case: a file renamed to a supported extension whose content isn't
# even that broad a file family at all (extension alone previously
# trusted this completely).
_MAGIC_BYTES: dict[DocumentType, tuple[bytes, ...]] = {
    DocumentType.PDF: (b"%PDF-",),
    # DOCX is a zip archive (local file header signature); this also
    # matches other zip-based formats, but that's fine — the point is
    # rejecting content that isn't even a zip archive at all.
    DocumentType.DOCX: (b"PK\x03\x04",),
}


def verify_content_matches_type(document_type: DocumentType, content: bytes) -> None:
    signatures = _MAGIC_BYTES.get(document_type)
    if signatures is None:
        return
    if not any(content.startswith(sig) for sig in signatures):
        raise InvalidDocumentError(
            f"file content does not match its {document_type.value} extension",
        )


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
    document_type = detect_document_type(safe_name)  # raises UnsupportedFileTypeError if unknown

    if len(content) == 0:
        raise InvalidDocumentError("uploaded file is empty", details={"filename": safe_name})
    if len(content) > max_size_bytes:
        raise InvalidDocumentError(
            f"uploaded file exceeds max size of {max_size_bytes} bytes",
            details={"filename": safe_name, "size": len(content)},
        )
    verify_content_matches_type(document_type, content)
    return safe_name
