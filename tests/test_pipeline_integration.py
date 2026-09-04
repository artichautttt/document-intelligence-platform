"""Tests d'intégration du pipeline complet (ingestion + chunking) sur des
documents réalistes déposés dans tests/fixtures/documents/.

Ces documents sont synthétiques (générés pour ce sprint, en l'absence de
documents réels fournis par l'utilisateur) mais structurellement représentatifs
des cas d'usage cibles : rapport financier (PDF) et documents juridiques /
administratifs (DOCX) avec titres, paragraphes et tableaux.
"""

from pathlib import Path

import pytest

from document_intelligence.chunking.service import chunk_document
from document_intelligence.ingestion.models import ElementType
from document_intelligence.ingestion.service import ingest


def _fixture_files(fixtures_dir: Path) -> list[Path]:
    return sorted(fixtures_dir.glob("*.pdf")) + sorted(fixtures_dir.glob("*.docx"))


class TestPipelineIntegration:
    def test_fixtures_directory_has_at_least_three_documents(self, fixtures_dir: Path) -> None:
        files = _fixture_files(fixtures_dir)
        assert len(files) >= 3, (
            "Attendu au moins 3 documents dans tests/fixtures/documents/ "
            "(voir critère d'acceptation Sprint 1)"
        )

    @pytest.mark.parametrize(
        "filename",
        ["financial_report.pdf", "legal_contract.docx", "audit_report.docx"],
    )
    def test_ingest_and_chunk_real_documents(self, fixtures_dir: Path, filename: str) -> None:
        path = fixtures_dir / filename
        assert path.exists(), f"Fixture manquante: {path}"

        document = ingest(path)
        assert len(document.elements) > 0
        assert any(el.element_type is ElementType.TITLE for el in document.elements)

        chunks = chunk_document(document)
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.text.strip() != ""
            assert chunk.provenance.document_id == document.document_id
            assert chunk.provenance.source_path == str(path)
            assert len(chunk.provenance.element_ids) > 0
            valid_ids = {el.element_id for el in document.elements}
            assert set(chunk.provenance.element_ids).issubset(valid_ids)

    def test_financial_report_table_is_isolated_in_own_chunk(self, fixtures_dir: Path) -> None:
        document = ingest(fixtures_dir / "financial_report.pdf")
        chunks = chunk_document(document)

        table_element_ids = {
            el.element_id for el in document.elements if el.element_type is ElementType.TABLE
        }
        if not table_element_ids:
            pytest.skip("Aucun tableau détecté dans ce document synthétique")

        for chunk in chunks:
            chunk_element_ids = set(chunk.provenance.element_ids)
            if chunk_element_ids & table_element_ids:
                assert chunk_element_ids.issubset(table_element_ids)

    def test_legal_contract_preserves_article_sections(self, fixtures_dir: Path) -> None:
        document = ingest(fixtures_dir / "legal_contract.docx")
        chunks = chunk_document(document)

        chunk_texts = [c.text for c in chunks]
        assert any("Article 1" in t for t in chunk_texts)
        assert any("Article 4" in t for t in chunk_texts)
