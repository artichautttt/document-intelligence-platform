"""Intégration : /metrics persiste ses données entre deux démarrages de l'API."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from document_intelligence.api.app import create_app
from document_intelligence.core import config
from document_intelligence.observability import registry
from tests.api.conftest import FakeLLMClient


def _start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(
        config.settings, "chroma_persist_directory", str(tmp_path / ".chroma")
    )
    monkeypatch.setattr(
        config.settings, "metrics_db_path", str(tmp_path / "metrics.db")
    )
    return TestClient(create_app(llm=FakeLLMClient()))


def test_metrics_survive_an_api_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, simple_docx: Path
) -> None:
    registry.reset()
    with _start(tmp_path, monkeypatch) as first:
        with simple_docx.open("rb") as f:
            first.post(
                "/documents",
                files={
                    "file": (
                        "sample.docx",
                        f,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
            )
        assert (
            first.post("/query", json={"query": "resultats", "k": 2}).status_code == 200
        )
        before = first.get("/metrics").json()

    registry.reset()  # simule un nouveau processus : plus rien en mémoire
    with _start(tmp_path, monkeypatch) as second:
        after = second.get("/metrics").json()

    assert before["persistent"] is True
    assert before["spans"]["api.query"]["count"] == 1
    assert after["spans"]["api.query"]["count"] == 1
    assert after["spans"]["node.retrieve"] == before["spans"]["node.retrieve"]
    registry.reset()
