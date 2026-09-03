import pytest

from agentic_rag.core.errors import InvalidDocumentError, UnsupportedFileTypeError
from agentic_rag.core.models import DocumentType
from agentic_rag.ingestion.loaders.validation import (
    detect_document_type,
    sanitize_filename,
    validate_upload,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("report.pdf", DocumentType.PDF),
        ("report.DOCX", DocumentType.DOCX),
        ("notes.txt", DocumentType.TXT),
        ("readme.md", DocumentType.MARKDOWN),
        ("page.html", DocumentType.HTML),
        ("data.csv", DocumentType.CSV),
        ("data.json", DocumentType.JSON),
    ],
)
def test_detect_document_type(filename: str, expected: DocumentType) -> None:
    assert detect_document_type(filename) == expected


def test_detect_document_type_rejects_unknown_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        detect_document_type("malware.exe")


def test_sanitize_filename_strips_directory_components() -> None:
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\Windows\\evil.txt") == "evil.txt"


def test_sanitize_filename_rejects_empty_result() -> None:
    with pytest.raises(InvalidDocumentError):
        sanitize_filename("../")


def test_validate_upload_rejects_empty_content() -> None:
    with pytest.raises(InvalidDocumentError):
        validate_upload("empty.txt", b"", max_size_bytes=1000)


def test_validate_upload_rejects_oversized_content() -> None:
    with pytest.raises(InvalidDocumentError):
        validate_upload("big.txt", b"x" * 100, max_size_bytes=10)


def test_validate_upload_accepts_valid_file() -> None:
    assert validate_upload("notes.txt", b"hello", max_size_bytes=1000) == "notes.txt"
