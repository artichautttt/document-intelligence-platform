"""Test d'intégration bout en bout : ingestion -> chunking -> vectorisation -> recherche.

Vérifie que la traçabilité chunk -> élément source survit à toute la chaîne,
y compris après un aller-retour par ChromaDB.
"""

from pathlib import Path

from document_intelligence.chunking.service import chunk_document
from document_intelligence.ingestion.service import ingest
from document_intelligence.vectorization.chroma_store import ChromaStore


class TestVectorizationPipelineIntegration:
    def test_ingest_chunk_vectorize_and_query(
        self, fixtures_dir: Path, tmp_path: Path
    ) -> None:
        document = ingest(fixtures_dir / "legal_contract.docx")
        chunks = chunk_document(document)

        store = ChromaStore(persist_directory=str(tmp_path / ".chroma"))
        store.add_chunks(chunks)

        results = store.query("resiliation du contrat en cas de manquement", k=3)

        assert len(results) > 0
        top_result = results[0]
        assert top_result.provenance.document_id == document.document_id
        assert top_result.provenance.source_path == str(
            fixtures_dir / "legal_contract.docx"
        )

        valid_element_ids = {el.element_id for el in document.elements}
        for result in results:
            assert set(result.provenance.element_ids).issubset(valid_element_ids)
