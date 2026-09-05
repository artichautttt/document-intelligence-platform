"""Tests de l'authentification par clé d'API (en-tête `X-API-Key`)."""

from fastapi.testclient import TestClient


def test_documents_rejects_request_without_api_key(client_with_api_key: TestClient) -> None:
    response = client_with_api_key.post(
        "/documents", files={"file": ("notes.txt", b"contenu", "text/plain")}
    )

    assert response.status_code == 401


def test_documents_rejects_request_with_wrong_api_key(client_with_api_key: TestClient) -> None:
    response = client_with_api_key.post(
        "/documents",
        files={"file": ("notes.txt", b"contenu", "text/plain")},
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401


def test_query_rejects_request_without_api_key(client_with_api_key: TestClient) -> None:
    response = client_with_api_key.post("/query", json={"query": "quelque chose"})

    assert response.status_code == 401


def test_query_accepts_request_with_correct_api_key(client_with_api_key: TestClient) -> None:
    response = client_with_api_key.post(
        "/query",
        json={"query": "quelque chose"},
        headers={"X-API-Key": "secret-key"},
    )

    # 404 attendu (aucun document ingere), mais l'authentification a laisse passer la requete.
    assert response.status_code == 404


def test_health_does_not_require_api_key(client_with_api_key: TestClient) -> None:
    response = client_with_api_key.get("/health")

    assert response.status_code == 200
