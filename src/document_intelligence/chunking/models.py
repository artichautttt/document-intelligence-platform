"""Modèles Pydantic représentant un chunk et sa provenance exacte."""

from pydantic import BaseModel


class ChunkProvenance(BaseModel):
    """Référence exacte vers les éléments source ayant produit un chunk.

    Cette traçabilité est le socle de l'exigence produit : toute réponse
    générée par le système RAG doit pouvoir être reliée aux chunks source
    exacts, et chaque chunk doit lui-même pouvoir être relié aux éléments
    du document original dont il est issu.
    """

    document_id: str
    source_path: str
    element_ids: list[str]


class Chunk(BaseModel):
    """Unité de texte destinée à être vectorisée, avec sa provenance complète."""

    chunk_id: str
    text: str
    provenance: ChunkProvenance
