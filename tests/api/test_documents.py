"""Tests de l'endpoint d'ingestion `/documents`."""

from pathlib import Path

from fastapi.testclient import TestClient


def test_upload_document_ingests_chunks_and_vectorizes(
    client: TestClient, simple_docx: Path
) -> None:
    with simple_docx.open("rb") as f:
        response = client.post(
            "/documents",
            files={"file": ("sample.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"]
    assert body["chunk_count"] >= 1


def test_upload_document_rejects_unsupported_format(client: TestClient, tmp_path: Path) -> None:
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("contenu quelconque")

    with unsupported.open("rb") as f:
        response = client.post("/documents", files={"file": ("notes.txt", f, "text/plain")})

    assert response.status_code == 415


def test_upload_document_rejects_corrupt_file(client: TestClient, corrupt_pdf: Path) -> None:
    with corrupt_pdf.open("rb") as f:
        response = client.post("/documents", files={"file": ("corrupt.pdf", f, "application/pdf")})

    assert response.status_code == 422
