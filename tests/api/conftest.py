"""Fixtures partagées pour les tests de l'API FastAPI."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from document_intelligence.api.app import create_app
from document_intelligence.core import config
from document_intelligence.orchestration.llm import LLMClient


class FakeLLMClient(LLMClient):
    """Client LLM déterministe utilisé pour éviter tout appel réseau en test."""

    def complete(self, prompt: str) -> str:
        return f"Réponse synthétique pour : {prompt}"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Client de test avec un ChromaStore isolé et un `LLMClient` factice branché."""
    monkeypatch.setattr(config.settings, "chroma_persist_directory", str(tmp_path / ".chroma"))
    app = create_app(llm=FakeLLMClient())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Client de test sans `LLMClient` configuré, pour vérifier le 503 de `/query`."""
    monkeypatch.setattr(config.settings, "chroma_persist_directory", str(tmp_path / ".chroma"))
    app = create_app(llm=None)
    with TestClient(app) as test_client:
        yield test_client
