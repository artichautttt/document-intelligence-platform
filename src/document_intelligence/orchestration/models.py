"""Modèles Pydantic représentant l'état et le résultat du pipeline d'agents."""

from pydantic import BaseModel

from document_intelligence.chunking.models import ChunkProvenance
from document_intelligence.vectorization.models import QueryResult


class RouteDecision(BaseModel):
    """Décision de l'agent de routage sur la manière de traiter une requête."""

    route: str
    k: int = 5


class Citation(BaseModel):
    """Lien entre un passage de la réponse générée et son chunk source exact.

    Chaque citation reporte la provenance jusqu'aux éléments du document
    original, condition nécessaire à la traçabilité exigée par le produit.
    """

    chunk_id: str
    provenance: ChunkProvenance
    score: float


class AnswerResult(BaseModel):
    """Réponse finale produite par le pipeline, avec ses citations source."""

    query: str
    answer: str
    citations: list[Citation]


class AgentState(BaseModel):
    """État transmis de nœud en nœud le long du pipeline d'orchestration."""

    query: str
    route: RouteDecision | None = None
    retrieved: list[QueryResult] = []
    answer: AnswerResult | None = None
