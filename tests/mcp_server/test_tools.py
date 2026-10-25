"""Appels directs des outils (`document_intelligence.tools`) et de `list_documents`."""

from pathlib import Path

import pytest

from document_intelligence import tools
from document_intelligence.chunking.models import Chunk
from document_intelligence.orchestration.backends import LangGraphBackend
from document_intelligence.orchestration.exceptions import EmptyQueryError
from document_intelligence.vectorization.chroma_store import ChromaStore
from document_intelligence.vectorization.hybrid_store import HybridStore
from document_intelligence.vectorization.models import QueryResult
from document_intelligence.vectorization.qdrant_store import QdrantStore
from document_intelligence.vectorization.store import VectorStore

from .conftest import CitingLLM, InMemoryStore, make_chunk


class TestSearchDocuments:
    def test_returns_hits_with_provenance_and_passes_k(
        self, store: InMemoryStore
    ) -> None:
        result = tools.search_documents(store, "redevance", k=2)

        assert store.queries == [("redevance", 2)]
        assert [h.source_path for h in result.hits] == ["/docs/contrat.pdf"] * 2
        assert result.hits[0].text.startswith("La redevance")
        assert result.hits[0].document_id == "doc-1"

    @pytest.mark.parametrize("k", [0, tools.MAX_K + 1])
    def test_rejects_out_of_range_k(self, store: InMemoryStore, k: int) -> None:
        with pytest.raises(ValueError, match="k doit"):
            tools.search_documents(store, "redevance", k=k)

    def test_rejects_empty_query(self, store: InMemoryStore) -> None:
        with pytest.raises(EmptyQueryError):
            tools.search_documents(store, "   ")


class TestAskQuestion:
    def test_runs_the_pipeline_and_returns_cited_sources(
        self, store: InMemoryStore
    ) -> None:
        llm = CitingLLM()
        answer = tools.ask_question(
            LangGraphBackend(store, llm), "Quel est le montant de la redevance ?", k=3
        )

        assert llm.calls == 1
        assert store.queries == [("Quel est le montant de la redevance ?", 3)]
        assert answer.sufficient_context is True
        assert answer.answer == "La redevance est de 184 000 euros HT."
        assert [s.chunk_id for s in answer.sources] == [store.chunks[0].chunk_id]
        assert answer.sources[0].source_path == "/docs/contrat.pdf"

    def test_without_llm_raises_explicit_error(self, store: InMemoryStore) -> None:
        with pytest.raises(tools.ToolUnavailableError, match="Aucun fournisseur LLM"):
            tools.ask_question(None, "Quel est le montant ?")


class TestListDocuments:
    def test_groups_chunks_by_document(self, store: InMemoryStore) -> None:
        documents = tools.list_documents(store).documents

        assert [(d.source_path, d.chunk_count) for d in documents] == [
            ("/docs/contrat.pdf", 2),
            ("/docs/rapport.pdf", 1),
        ]

    def test_store_without_listing_raises_not_implemented(self) -> None:
        class QueryOnlyStore(VectorStore):
            def add_chunks(self, chunks: list[Chunk]) -> None:
                pass

            def query(self, text: str, k: int = 5) -> list[QueryResult]:
                return []

        with pytest.raises(NotImplementedError):
            QueryOnlyStore().list_documents()


def _chunks() -> list[Chunk]:
    return [
        make_chunk("Premier passage du contrat.", "doc-1", "/docs/contrat.pdf"),
        make_chunk("Second passage du contrat.", "doc-1", "/docs/contrat.pdf"),
        make_chunk("Passage du rapport.", "doc-2", "/docs/rapport.pdf"),
    ]


class TestBackendsListDocuments:
    def test_chroma(self, tmp_path: Path) -> None:
        store = ChromaStore(persist_directory=str(tmp_path / ".chroma"))
        store.add_chunks(_chunks())
        assert [(d.document_id, d.chunk_count) for d in store.list_documents()] == [
            ("doc-1", 2),
            ("doc-2", 1),
        ]

    def test_chroma_empty_collection(self, tmp_path: Path) -> None:
        assert (
            ChromaStore(persist_directory=str(tmp_path / ".chroma")).list_documents()
            == []
        )

    def test_qdrant_in_memory(self) -> None:
        store = QdrantStore(location=":memory:")
        store.add_chunks(_chunks())
        assert [(d.document_id, d.chunk_count) for d in store.list_documents()] == [
            ("doc-1", 2),
            ("doc-2", 1),
        ]

    def test_hybrid_delegates_to_inner_store(self, tmp_path: Path) -> None:
        inner = ChromaStore(persist_directory=str(tmp_path / ".chroma"))
        inner.add_chunks(_chunks())
        assert HybridStore(inner).list_documents() == inner.list_documents()
