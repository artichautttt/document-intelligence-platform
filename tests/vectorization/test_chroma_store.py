"""Tests de l'implémentation ChromaDB de VectorStore."""

import uuid
from pathlib import Path

import pytest

from document_intelligence.chunking.models import Chunk, ChunkProvenance
from document_intelligence.vectorization.chroma_store import ChromaStore
from document_intelligence.vectorization.exceptions import EmptyChunkListError


def _chunk(text: str, document_id: str = "doc-1") -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        text=text,
        provenance=ChunkProvenance(
            document_id=document_id,
            source_path="/tmp/fake.pdf",
            element_ids=[str(uuid.uuid4())],
        ),
    )


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(persist_directory=str(tmp_path / ".chroma"))


class TestChromaStore:
    def test_add_chunks_raises_on_empty_list(self, store: ChromaStore) -> None:
        with pytest.raises(EmptyChunkListError):
            store.add_chunks([])

    def test_query_returns_most_similar_chunk_first(self, store: ChromaStore) -> None:
        chunks = [
            _chunk("Le chat dort paisiblement sur le canape du salon."),
            _chunk("Le chiffre d'affaires trimestriel a augmente de quinze pourcent."),
            _chunk("La marge operationnelle s'est amelioree grace a la baisse des couts."),
        ]
        store.add_chunks(chunks)

        results = store.query("Quels sont les resultats financiers du trimestre ?", k=2)

        assert len(results) == 2
        result_texts = [r.text for r in results]
        assert any("chiffre d'affaires" in t or "marge" in t for t in result_texts)
        assert "chat" not in result_texts[0]

    def test_query_result_preserves_provenance(self, store: ChromaStore) -> None:
        chunk = _chunk("Un paragraphe unique pour verifier la tracabilite.", document_id="doc-42")
        store.add_chunks([chunk])

        results = store.query("paragraphe tracabilite", k=1)

        assert len(results) == 1
        assert results[0].chunk_id == chunk.chunk_id
        assert results[0].provenance.document_id == "doc-42"
        assert results[0].provenance.element_ids == chunk.provenance.element_ids

    def test_add_chunks_is_idempotent_via_upsert(self, store: ChromaStore) -> None:
        chunk = _chunk("Texte stable pour verifier l'upsert.")
        store.add_chunks([chunk])
        store.add_chunks([chunk])

        results = store.query("texte stable upsert", k=5)

        matching = [r for r in results if r.chunk_id == chunk.chunk_id]
        assert len(matching) == 1
