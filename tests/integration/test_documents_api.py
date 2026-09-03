import uuid

import pytest


@pytest.mark.asyncio
async def test_upload_document_end_to_end(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    assert collection_resp.status_code == 201
    collection_id = collection_resp.json()["id"]

    upload_resp = await client.post(
        "/documents",
        data={"collection_id": collection_id},
        files={"file": ("notes.txt", b"Paragraph one.\n\nParagraph two.", "text/plain")},
    )
    assert upload_resp.status_code == 201
    body = upload_resp.json()
    assert body["document"]["filename"] == "notes.txt"
    assert body["document"]["document_type"] == "txt"
    assert body["version"]["version_number"] == 1
    assert body["element_count"] == 2

    document_id = body["document"]["id"]
    get_resp = await client.get(f"/documents/{document_id}")
    assert get_resp.status_code == 200
    assert len(get_resp.json()["versions"]) == 1

    list_resp = await client.get("/documents", params={"collection_id": collection_id})
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1


@pytest.mark.asyncio
async def test_get_document_not_found_returns_404(client) -> None:
    response = await client.get(f"/documents/{uuid.uuid4()}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_upload_unsupported_file_type_returns_415(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]

    response = await client.post(
        "/documents",
        data={"collection_id": collection_id},
        files={"file": ("malware.exe", b"binary content", "application/octet-stream")},
    )
    assert response.status_code == 415
    assert response.json()["code"] == "UNSUPPORTED_FILE_TYPE"


@pytest.mark.asyncio
async def test_upload_empty_file_returns_400(client) -> None:
    collection_resp = await client.post(
        "/collections", json={"name": f"col-{uuid.uuid4().hex[:8]}"}
    )
    collection_id = collection_resp.json()["id"]

    response = await client.post(
        "/documents",
        data={"collection_id": collection_id},
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_DOCUMENT"
