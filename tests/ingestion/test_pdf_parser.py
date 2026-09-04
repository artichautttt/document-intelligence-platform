"""Tests du parser PDF."""

from pathlib import Path

import pytest

from document_intelligence.ingestion.exceptions import CorruptFileError
from document_intelligence.ingestion.parsers.pdf_parser import PdfParser


class TestPdfParser:
    def test_supports_pdf_extension(self) -> None:
        parser = PdfParser()
        assert parser.supports(Path("report.pdf")) is True
        assert parser.supports(Path("report.docx")) is False

    def test_parses_pdf_into_elements(self, simple_pdf: Path) -> None:
        parser = PdfParser()

        document = parser.parse(simple_pdf)

        assert document.source_format == "pdf"
        assert document.source_path == str(simple_pdf)
        assert len(document.elements) > 0
        full_text = " ".join(el.text for el in document.elements)
        assert "Rapport Annuel" in full_text

    def test_raises_corrupt_file_error_on_invalid_pdf(self, corrupt_pdf: Path) -> None:
        parser = PdfParser()

        with pytest.raises(CorruptFileError):
            parser.parse(corrupt_pdf)
