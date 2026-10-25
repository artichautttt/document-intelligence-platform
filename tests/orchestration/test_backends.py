"""Sélection et comportement des backends RAG (`RAG_BACKEND`)."""

import importlib.util

import pytest

from document_intelligence.core.config import settings
from document_intelligence.orchestration.backends import (
    LangGraphBackend,
    check_backend,
)
from document_intelligence.services import build_backend

from ..mcp_server.conftest import CitingLLM, InMemoryStore, demo_store


@pytest.fixture
def store() -> InMemoryStore:
    return demo_store()


def test_langgraph_backend_runs_the_graph(store: InMemoryStore) -> None:
    llm = CitingLLM()

    result = LangGraphBackend(store, llm).answer("Montant de la redevance ?", k=2)

    assert llm.calls == 1
    assert store.queries == [("Montant de la redevance ?", 2)]
    assert result.sufficient_context is True
    assert [c.chunk_id for c in result.citations] == [store.chunks[0].chunk_id]


@pytest.mark.parametrize(
    ("raw", "expected"), [("langgraph", "langgraph"), (" LangChain ", "langchain")]
)
def test_check_backend_normalizes(raw: str, expected: str) -> None:
    assert check_backend(raw) == expected


def test_check_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="non supporté"):
        check_backend("llamaindex")


class TestBuildBackend:
    def test_langgraph_uses_the_injected_llm(
        self, store: InMemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "rag_backend", "langgraph")
        llm = CitingLLM()

        backend = build_backend(store, llm)

        assert isinstance(backend, LangGraphBackend)
        assert backend.llm is llm

    def test_langgraph_without_llm_is_none(
        self, store: InMemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "rag_backend", "langgraph")
        assert build_backend(store, None) is None

    def test_langchain_without_key_is_none(
        self, store: InMemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "rag_backend", "langchain")
        monkeypatch.setattr(settings, "anthropic_api_key", None)
        assert build_backend(store, CitingLLM()) is None

    @pytest.mark.skipif(
        importlib.util.find_spec("langchain_anthropic") is None,
        reason="groupe optionnel langchain non installé",
    )
    def test_langchain_builds_the_lcel_chain_without_calling_the_api(
        self, store: InMemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "rag_backend", "langchain")
        monkeypatch.setattr(settings, "llm_provider", "anthropic")
        monkeypatch.setattr(settings, "anthropic_api_key", "cle-de-test")

        backend = build_backend(store, None)

        assert backend is not None
        assert backend.name == "langchain"

    def test_unknown_backend_raises(
        self, store: InMemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "rag_backend", "inconnu")
        with pytest.raises(ValueError):
            build_backend(store, CitingLLM())
