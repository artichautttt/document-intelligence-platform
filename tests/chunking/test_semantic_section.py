"""Tests de la stratégie de chunking structurel par sections."""

import uuid

from document_intelligence.chunking.strategies.semantic_section import (
    SemanticSectionChunker,
)
from document_intelligence.ingestion.models import (
    DocumentElement,
    ElementType,
    ParsedDocument,
)


def _element(element_type: ElementType, text: str) -> DocumentElement:
    return DocumentElement(
        element_id=str(uuid.uuid4()), element_type=element_type, text=text
    )


def _document(elements: list[DocumentElement]) -> ParsedDocument:
    return ParsedDocument(
        document_id=str(uuid.uuid4()),
        source_path="/tmp/fake.pdf",
        source_format="pdf",
        elements=elements,
    )


class TestSemanticSectionChunker:
    def test_respects_section_boundaries(self) -> None:
        elements = [
            _element(ElementType.TITLE, "Section A"),
            _element(ElementType.NARRATIVE_TEXT, "Contenu de la section A."),
            _element(ElementType.TITLE, "Section B"),
            _element(ElementType.NARRATIVE_TEXT, "Contenu de la section B."),
        ]
        chunker = SemanticSectionChunker(max_chunk_chars=1000)

        chunks = chunker.chunk(_document(elements))

        assert len(chunks) == 2
        assert "Section A" in chunks[0].text
        assert "Section B" not in chunks[0].text
        assert "Section B" in chunks[1].text

    def test_keeps_tables_as_dedicated_chunks(self) -> None:
        elements = [
            _element(ElementType.TITLE, "Résultats"),
            _element(ElementType.NARRATIVE_TEXT, "Texte avant le tableau."),
            _element(ElementType.TABLE, "Q1 | 1200000"),
            _element(ElementType.NARRATIVE_TEXT, "Texte après le tableau."),
        ]
        chunker = SemanticSectionChunker(max_chunk_chars=1000)

        chunks = chunker.chunk(_document(elements))

        table_chunks = [c for c in chunks if c.text == "Q1 | 1200000"]
        assert len(table_chunks) == 1
        assert table_chunks[0].provenance.element_ids == [elements[2].element_id]

    def test_splits_oversized_section_on_sentence_boundaries(self) -> None:
        long_sentence_a = "Phrase numero un qui decrit un point important. " * 5
        long_sentence_b = "Phrase numero deux qui decrit un autre point important. " * 5
        elements = [
            _element(ElementType.TITLE, "Analyse"),
            _element(ElementType.NARRATIVE_TEXT, long_sentence_a + long_sentence_b),
        ]
        chunker = SemanticSectionChunker(max_chunk_chars=200)

        chunks = chunker.chunk(_document(elements))

        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 250
            assert not c.text.strip().endswith(("numero un qui", "numero deux qui"))

    def test_chunk_provenance_references_source_elements(self) -> None:
        elements = [
            _element(ElementType.TITLE, "Contexte"),
            _element(ElementType.NARRATIVE_TEXT, "Un paragraphe court."),
        ]
        document = _document(elements)
        chunker = SemanticSectionChunker(max_chunk_chars=1000)

        chunks = chunker.chunk(document)

        assert len(chunks) == 1
        assert chunks[0].provenance.document_id == document.document_id
        assert chunks[0].provenance.source_path == document.source_path
        assert set(chunks[0].provenance.element_ids) == {
            el.element_id for el in elements
        }

    def test_empty_document_produces_no_chunks(self) -> None:
        chunker = SemanticSectionChunker(max_chunk_chars=1000)

        chunks = chunker.chunk(_document([]))

        assert chunks == []
