"""Tests du pipeline d'orchestration (routage -> retrieval -> synthèse)."""

import uuid

import pytest

from document_intelligence.chunking.models import Chunk, ChunkProvenance
from document_intelligence.orchestration.exceptions import EmptyQueryError, NoRelevantChunkError
from document_intelligence.orchestration.graph import answer_query
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.vectorization.exceptions import EmptyChunkListError
from document_intelligence.vectorization.models import QueryResult
from document_intelligence.vectorization.store import VectorStore


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        text=text,
        provenance=ChunkProvenance(
            document_id="doc-1", source_path="/tmp/fake.pdf", element_ids=[str(uuid.uuid4())]
        ),
    )


class FakeVectorStore(VectorStore):
    """Vector store en mémoire retournant ses chunks dans l'ordre d'insertion."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise EmptyChunkListError("La liste de chunks est vide.")
        self._chunks.extend(chunks)

    def query(self, text: str, k: int = 5) -> list[QueryResult]:
        return [
            QueryResult(chunk_id=c.chunk_id, text=c.text, score=1.0, provenance=c.provenance)
            for c in self._chunks[:k]
        ]


class EmptyVectorStore(VectorStore):
    """Vector store qui ne retourne jamais aucun résultat."""

    def add_chunks(self, chunks: list[Chunk]) -> None:
        pass

    def query(self, text: str, k: int = 5) -> list[QueryResult]:
        return []


class FakeLLMClient(LLMClient):
    """Client LLM déterministe qui renvoie le prompt reçu, pour vérification en test."""

    def complete(self, prompt: str) -> str:
        return f"Réponse synthétique pour : {prompt}"


@pytest.fixture
def store() -> FakeVectorStore:
    fake_store = FakeVectorStore()
    fake_store.add_chunks(
        [
            _chunk("Le chiffre d'affaires trimestriel a augmente de quinze pourcent."),
            _chunk("La marge operationnelle s'est amelioree grace a la baisse des couts."),
        ]
    )
    return fake_store


@pytest.fixture
def llm() -> FakeLLMClient:
    return FakeLLMClient()


class TestAnswerQuery:
    def test_answer_query_returns_citations_for_retrieved_chunks(
        self, store: FakeVectorStore, llm: FakeLLMClient
    ) -> None:
        result = answer_query("Quels sont les resultats financiers ?", store=store, llm=llm, k=2)

        assert result.query == "Quels sont les resultats financiers ?"
        assert len(result.citations) == 2
        assert result.citations[0].provenance.document_id == "doc-1"
        assert "Réponse synthétique" in result.answer

    def test_answer_query_raises_on_empty_query(
        self, store: FakeVectorStore, llm: FakeLLMClient
    ) -> None:
        with pytest.raises(EmptyQueryError):
            answer_query("   ", store=store, llm=llm)

    def test_answer_query_raises_when_no_chunk_found(self, llm: FakeLLMClient) -> None:
        with pytest.raises(NoRelevantChunkError):
            answer_query("question sans contexte", store=EmptyVectorStore(), llm=llm)
