"""Tests du parser DOCX."""

from pathlib import Path

import pytest

from document_intelligence.ingestion.exceptions import CorruptFileError
from document_intelligence.ingestion.models import ElementType
from document_intelligence.ingestion.parsers.docx_parser import DocxParser


class TestDocxParser:
    def test_supports_docx_extension(self) -> None:
        parser = DocxParser()
        assert parser.supports(Path("report.docx")) is True
        assert parser.supports(Path("report.pdf")) is False

    def test_parses_titles_paragraphs_and_tables(self, simple_docx: Path) -> None:
        parser = DocxParser()

        document = parser.parse(simple_docx)

        assert document.source_format == "docx"
        assert document.source_path == str(simple_docx)
        element_types = [el.element_type for el in document.elements]
        assert ElementType.TITLE in element_types
        assert ElementType.TABLE in element_types

    def test_preserves_element_order(self, simple_docx: Path) -> None:
        parser = DocxParser()

        document = parser.parse(simple_docx)

        titles = [
            el.text for el in document.elements if el.element_type is ElementType.TITLE
        ]
        assert titles == ["Introduction", "Résultats financiers"]

    def test_raises_corrupt_file_error_on_invalid_docx(
        self, corrupt_docx: Path
    ) -> None:
        parser = DocxParser()

        with pytest.raises(CorruptFileError):
            parser.parse(corrupt_docx)
