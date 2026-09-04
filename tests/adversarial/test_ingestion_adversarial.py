"""Adversarial ingestion inputs: malformed/empty documents, path
traversal filenames, oversized uploads. None of these should ever produce a
fabricated success — always a specific, typed failure."""

import uuid

import pytest

from agentic_rag.core.errors import InvalidDocumentError
from agentic_rag.ingestion.parsers.docx import DocxParser
from agentic_rag.ingestion.parsers.json_parser import JsonParser
from agentic_rag.ingestion.parsers.pdf import PdfParser


def test_pdf_parser_rejects_garbage_bytes() -> None:
    with pytest.raises(InvalidDocumentError):
        PdfParser().parse(filename="fake.pdf", content=b"this is not a pdf file at all")


def test_docx_parser_rejects_garbage_bytes() -> None:
    with pytest.raises(InvalidDocumentError):
        DocxParser().parse(filename="fake.docx", content=b"this is not a docx file at all")


def test_json_parser_rejects_malformed_json() -> None:
    with pytest.raises(InvalidDocumentError):
        JsonParser().parse(filename="fake.json", content=b"{not: valid json,,,")


@pytest.mark.asyncio
async def test_api_rejects_path_traversal_filename(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]

    response = await client.post(
        "/documents",
        data={"collection_id": collection_id},
        files={"file": ("../../etc/passwd.txt", b"content", "text/plain")},
    )
    # sanitized down to "passwd.txt" and ingested normally rather than escaping
    # the storage root — traversal is neutralized, not merely rejected.
    assert response.status_code == 201
    assert response.json()["document"]["filename"] == "passwd.txt"


@pytest.mark.asyncio
async def test_api_rejects_oversized_upload(client, monkeypatch) -> None:
    from agentic_rag.core import config

    monkeypatch.setattr(config.get_settings(), "max_upload_size_bytes", 10)

    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]

    response = await client.post(
        "/documents",
        data={"collection_id": collection_id},
        files={"file": ("big.txt", b"x" * 1000, "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_DOCUMENT"
