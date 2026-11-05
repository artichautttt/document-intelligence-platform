"""Tests de l'endpoint d'ingestion `/documents`."""

from pathlib import Path

from fastapi.testclient import TestClient


def test_upload_document_ingests_chunks_and_vectorizes(
    client: TestClient, simple_docx: Path
) -> None:
    with simple_docx.open("rb") as f:
        response = client.post(
            "/documents",
            files={
                "file": (
                    "sample.docx",
                    f,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"]
    assert body["chunk_count"] >= 1


def test_upload_document_rejects_unsupported_format(
    client: TestClient, tmp_path: Path
) -> None:
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("contenu quelconque")

    with unsupported.open("rb") as f:
        response = client.post(
            "/documents", files={"file": ("notes.txt", f, "text/plain")}
        )

    assert response.status_code == 415


def test_upload_document_rejects_corrupt_file(
    client: TestClient, corrupt_pdf: Path
) -> None:
    with corrupt_pdf.open("rb") as f:
        response = client.post(
            "/documents", files={"file": ("corrupt.pdf", f, "application/pdf")}
        )

    assert response.status_code == 422


def test_upload_bpmn_gives_one_chunk_per_task(client: TestClient) -> None:
    path = Path("examples/bpmn/exemple_note_de_frais.bpmn")
    with path.open("rb") as f:
        response = client.post(
            "/documents", files={"file": (path.name, f, "application/xml")}
        )

    assert response.status_code == 201
    assert response.json()["chunk_count"] == 6


def test_upload_bpmn_with_a_dtd_is_rejected(client: TestClient, tmp_path: Path) -> None:
    path = tmp_path / "entites.bpmn"
    path.write_bytes(
        b'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY e "x">]><definitions/>'
    )
    with path.open("rb") as f:
        response = client.post(
            "/documents", files={"file": (path.name, f, "application/xml")}
        )

    assert response.status_code == 422
