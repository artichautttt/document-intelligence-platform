"""Tests du endpoint de santé."""

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_exposes_pipeline_and_llm_stats(client: TestClient) -> None:
    body = client.get("/metrics").json()

    assert set(body) == {"spans", "llm", "persistent"}
    assert body["llm"]["pricing_configured"] is False
