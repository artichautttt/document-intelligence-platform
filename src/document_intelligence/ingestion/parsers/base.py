"""Interface abstraite commune à tous les parsers de documents."""

from abc import ABC, abstractmethod
from pathlib import Path

from document_intelligence.ingestion.models import ParsedDocument


class DocumentParser(ABC):
    """Contrat que doit respecter tout parser capable de convertir un fichier en `ParsedDocument`.

    Cette abstraction permet d'ajouter de nouveaux formats (HTML, PPTX, images
    scannées via OCR...) sans modifier le reste du pipeline d'ingestion.
    """

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Indique si ce parser sait traiter le fichier donné (généralement basé sur l'extension)."""

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """Parse le fichier et retourne sa représentation structurée.

        Raises:
            CorruptFileError: si le fichier est illisible ou malformé.
            EmptyDocumentError: si aucun contenu exploitable n'a pu être extrait.
        """
