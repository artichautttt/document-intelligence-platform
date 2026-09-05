"""Point d'entrée public du module d'ingestion."""

from pathlib import Path

from document_intelligence.core.logging import get_logger
from document_intelligence.ingestion.exceptions import UnsupportedFormatError
from document_intelligence.ingestion.models import ParsedDocument
from document_intelligence.ingestion.parsers.base import DocumentParser
from document_intelligence.ingestion.parsers.docx_parser import DocxParser
from document_intelligence.ingestion.parsers.pdf_parser import PdfParser

logger = get_logger(__name__)

_DEFAULT_PARSERS: list[DocumentParser] = [PdfParser(), DocxParser()]


def ingest(
    path: str | Path, parsers: list[DocumentParser] | None = None
) -> ParsedDocument:
    """Parse un fichier en sélectionnant automatiquement le parser adapté à son format.

    Args:
        path: chemin vers le document à ingérer.
        parsers: liste de parsers disponibles (par défaut PDF et DOCX). Permet
            l'injection de dépendances pour les tests ou l'ajout de formats.

    Raises:
        UnsupportedFormatError: si aucun parser ne supporte le format du fichier.
        CorruptFileError: si le fichier est illisible ou malformé.
        EmptyDocumentError: si aucun contenu exploitable n'a pu être extrait.
    """
    resolved_path = Path(path)
    active_parsers = parsers if parsers is not None else _DEFAULT_PARSERS

    for parser in active_parsers:
        if parser.supports(resolved_path):
            logger.info(
                "ingestion.start", path=str(resolved_path), parser=type(parser).__name__
            )
            document = parser.parse(resolved_path)
            logger.info(
                "ingestion.success",
                path=str(resolved_path),
                document_id=document.document_id,
                element_count=len(document.elements),
            )
            return document

    raise UnsupportedFormatError(
        f"Aucun parser disponible pour le format du fichier '{resolved_path}'"
    )
