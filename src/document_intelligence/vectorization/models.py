"""Modèles Pydantic pour les résultats de recherche vectorielle."""

from pydantic import BaseModel

from document_intelligence.chunking.models import ChunkProvenance


class QueryResult(BaseModel):
    """Un chunk retrouvé par recherche de similarité, avec son score et sa provenance.

    Conserver la provenance jusque dans le résultat de recherche garantit que
    toute réponse générée en aval par les agents pourra être reliée au(x)
    chunk(s) source(s) exact(s), comme l'exige la traçabilité du système.
    """

    chunk_id: str
    text: str
    score: float
    provenance: ChunkProvenance
