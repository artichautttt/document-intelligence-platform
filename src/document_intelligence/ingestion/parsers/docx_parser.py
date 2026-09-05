"""Parser DOCX basé sur `unstructured`, préservant titres, tableaux et sections.

Voir `pdf_parser.py` pour la justification du choix de `unstructured`.
"""

import uuid
from pathlib import Path

from unstructured.partition.docx import partition_docx

from document_intelligence.ingestion.exceptions import (
    CorruptFileError,
    EmptyDocumentError,
)
from document_intelligence.ingestion.models import (
    DocumentElement,
    ElementType,
    ParsedDocument,
)
from document_intelligence.ingestion.parsers.base import DocumentParser

_TYPE_MAP: dict[str, ElementType] = {
    "Title": ElementType.TITLE,
    "NarrativeText": ElementType.NARRATIVE_TEXT,
    "Table": ElementType.TABLE,
    "ListItem": ElementType.LIST_ITEM,
}


class DocxParser(DocumentParser):
    """Parser pour les fichiers `.docx`."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".docx"

    def parse(self, path: Path) -> ParsedDocument:
        try:
            raw_elements = partition_docx(filename=str(path))
        except Exception as exc:
            raise CorruptFileError(
                f"Impossible de parser le DOCX '{path}': {exc}"
            ) from exc

        elements = [
            DocumentElement(
                element_id=str(uuid.uuid4()),
                element_type=_TYPE_MAP.get(type(el).__name__, ElementType.OTHER),
                text=str(el).strip(),
                page_number=getattr(el.metadata, "page_number", None),
            )
            for el in raw_elements
            if str(el).strip()
        ]

        if not elements:
            raise EmptyDocumentError(
                f"Aucun contenu exploitable extrait du DOCX '{path}'"
            )

        return ParsedDocument(
            document_id=str(uuid.uuid4()),
            source_path=str(path),
            source_format="docx",
            elements=elements,
        )
