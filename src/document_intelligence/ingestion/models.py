"""Modèles Pydantic représentant un document parsé et sa structure interne."""

from enum import Enum

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    """Type structurel d'un élément extrait d'un document source."""

    TITLE = "title"
    NARRATIVE_TEXT = "narrative_text"
    TABLE = "table"
    LIST_ITEM = "list_item"
    OTHER = "other"


class DocumentElement(BaseModel):
    """Un élément structurel unique extrait du document (titre, paragraphe, tableau...)."""

    element_id: str
    element_type: ElementType
    text: str
    page_number: int | None = None


class ParsedDocument(BaseModel):
    """Représentation structurée et traçable d'un document source parsé.

    Chaque `DocumentElement` conserve l'ordre d'apparition dans le document,
    ce qui permet au module de chunking de reconstituer les frontières de
    sections sans jamais retourner au fichier source.
    """

    document_id: str
    source_path: str
    source_format: str
    elements: list[DocumentElement] = Field(default_factory=list)
