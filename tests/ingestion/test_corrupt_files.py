"""Tests de robustesse du service d'ingestion face aux fichiers corrompus ou non supportés."""

from pathlib import Path

import pytest

from document_intelligence.ingestion.exceptions import (
    CorruptFileError,
    UnsupportedFormatError,
)
from document_intelligence.ingestion.service import ingest


class TestIngestionRobustness:
    def test_raises_unsupported_format_error_for_unknown_extension(
        self, tmp_path: Path
    ) -> None:
        unknown_file = tmp_path / "notes.txt"
        unknown_file.write_text("some content")

        with pytest.raises(UnsupportedFormatError):
            ingest(unknown_file)

    def test_raises_corrupt_file_error_and_does_not_crash_silently(
        self, corrupt_pdf: Path
    ) -> None:
        with pytest.raises(CorruptFileError):
            ingest(corrupt_pdf)

    def test_ingest_routes_docx_to_docx_parser(self, simple_docx: Path) -> None:
        document = ingest(simple_docx)

        assert document.source_format == "docx"
        assert len(document.elements) > 0
