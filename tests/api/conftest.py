"""Fixtures partagées pour les tests de l'API FastAPI."""

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from document_intelligence.api.app import create_app
from document_intelligence.core import config
from document_intelligence.orchestration.llm import LLMClient, T


class FakeLLMClient(LLMClient):
    """Client LLM déterministe : cite tous les extraits reçus dans son prompt."""

    def complete(self, prompt: str) -> str:
        return f"Réponse synthétique pour : {prompt}"

    def complete_structured(
        self, prompt: str, schema: type[T], system: str | None = None
    ) -> T:
        chunk_ids = re.findall(r'<extrait id="([^"]+)"', prompt)
        return schema.model_validate(
            {
                "answer": f"Réponse synthétique pour : {prompt}",
                "cited_chunk_ids": chunk_ids,
                "sufficient_context": True,
            }
        )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Client de test avec un ChromaStore isolé et un `LLMClient` factice branché."""
    monkeypatch.setattr(
        config.settings, "chroma_persist_directory", str(tmp_path / ".chroma")
    )
    app = create_app(llm=FakeLLMClient())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Client de test sans `LLMClient` configuré, pour vérifier le 503 de `/query`."""
    monkeypatch.setattr(
        config.settings, "chroma_persist_directory", str(tmp_path / ".chroma")
    )
    # Sans cela, une clé présente dans le `.env` local ferait construire un vrai client.
    monkeypatch.setattr(config.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(config.settings, "anthropic_api_key", None)
    app = create_app(llm=None)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_with_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Client de test avec une clé d'API requise (`settings.api_key` défini)."""
    monkeypatch.setattr(
        config.settings, "chroma_persist_directory", str(tmp_path / ".chroma")
    )
    monkeypatch.setattr(config.settings, "api_key", "secret-key")
    app = create_app(llm=FakeLLMClient())
    with TestClient(app) as test_client:
        yield test_client
