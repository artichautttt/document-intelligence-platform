"""Parser PDF basé sur `pdfplumber`, préservant titres, tableaux et sections.

`pdfplumber` a été choisi plutôt que `unstructured[pdf]` (qui embarque un
modèle d'inférence de layout basé sur torch/opencv, plusieurs centaines de Mo)
pour garder une empreinte disque et mémoire minimale, adaptée à un pipeline
d'ingestion conteneurisé. `pdfplumber` expose la taille de police par mot, ce
qui permet une détection heuristique fiable des titres (police significativement
plus grande que la médiane de la page), et une extraction native des tableaux
via sa détection de lignes/colonnes.
"""

import statistics
import uuid
from pathlib import Path
from typing import Any

import pdfplumber
from pdfplumber.page import Page

from document_intelligence.ingestion.exceptions import CorruptFileError, EmptyDocumentError
from document_intelligence.ingestion.models import DocumentElement, ElementType, ParsedDocument
from document_intelligence.ingestion.parsers.base import DocumentParser

_TITLE_SIZE_RATIO = 1.15


class PdfParser(DocumentParser):
    """Parser pour les fichiers `.pdf`."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def parse(self, path: Path) -> ParsedDocument:
        try:
            elements = self._extract_elements(path)
        except CorruptFileError:
            raise
        except Exception as exc:  # pdfminer/pdfplumber lèvent des exceptions variées
            raise CorruptFileError(f"Impossible de parser le PDF '{path}': {exc}") from exc

        if not elements:
            raise EmptyDocumentError(f"Aucun contenu exploitable extrait du PDF '{path}'")

        return ParsedDocument(
            document_id=str(uuid.uuid4()),
            source_path=str(path),
            source_format="pdf",
            elements=elements,
        )

    def _extract_elements(self, path: Path) -> list[DocumentElement]:
        elements: list[DocumentElement] = []

        try:
            pdf = pdfplumber.open(str(path))
        except Exception as exc:
            raise CorruptFileError(f"Fichier PDF invalide '{path}': {exc}") from exc

        with pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                elements.extend(self._extract_text_lines(page, page_number))
                elements.extend(self._extract_tables(page, page_number))

        return elements

    def _extract_text_lines(self, page: Page, page_number: int) -> list[DocumentElement]:
        words = page.extract_words(extra_attrs=["size"])
        if not words:
            return []

        median_size = statistics.median(w["size"] for w in words)

        lines: dict[float, list[dict[str, Any]]] = {}
        for word in words:
            key = round(word["top"], 1)
            lines.setdefault(key, []).append(word)

        elements: list[DocumentElement] = []
        for top in sorted(lines):
            line_words = sorted(lines[top], key=lambda w: w["x0"])
            text = " ".join(w["text"] for w in line_words).strip()
            if not text:
                continue

            avg_size = sum(w["size"] for w in line_words) / len(line_words)
            element_type = (
                ElementType.TITLE if avg_size > median_size * _TITLE_SIZE_RATIO
                else ElementType.NARRATIVE_TEXT
            )
            elements.append(
                DocumentElement(
                    element_id=str(uuid.uuid4()),
                    element_type=element_type,
                    text=text,
                    page_number=page_number,
                )
            )

        return elements

    def _extract_tables(self, page: Page, page_number: int) -> list[DocumentElement]:
        elements: list[DocumentElement] = []
        for table in page.extract_tables():
            rows = [
                " | ".join(cell.strip() if cell else "" for cell in row)
                for row in table
                if row
            ]
            text = "\n".join(rows).strip()
            if not text:
                continue
            elements.append(
                DocumentElement(
                    element_id=str(uuid.uuid4()),
                    element_type=ElementType.TABLE,
                    text=text,
                    page_number=page_number,
                )
            )
        return elements
