"""Tests de l'endpoint de recherche/synthèse `/query`."""

from pathlib import Path

from fastapi.testclient import TestClient


def _upload(client: TestClient, simple_docx: Path) -> None:
    with simple_docx.open("rb") as f:
        response = client.post(
            "/documents",
            files={"file": ("sample.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
    assert response.status_code == 201


def test_query_returns_citations_for_ingested_document(
    client: TestClient, simple_docx: Path
) -> None:
    _upload(client, simple_docx)

    response = client.post("/query", json={"query": "resultats financiers", "k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "resultats financiers"
    assert len(body["citations"]) >= 1
    assert "Réponse synthétique" in body["answer"]


def test_query_rejects_empty_query(client: TestClient, simple_docx: Path) -> None:
    _upload(client, simple_docx)

    response = client.post("/query", json={"query": "   "})

    assert response.status_code == 400


def test_query_returns_404_when_no_document_ingested(client: TestClient) -> None:
    response = client.post("/query", json={"query": "quelque chose"})

    assert response.status_code == 404


def test_query_returns_503_when_llm_not_configured(client_without_llm: TestClient) -> None:
    response = client_without_llm.post("/query", json={"query": "quelque chose"})

    assert response.status_code == 503
