"""Modèles Pydantic représentant l'état et le résultat du pipeline d'agents."""

from pydantic import BaseModel, Field

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


class GroundedAnswer(BaseModel):
    """Sortie structurée attendue du LLM à l'étape de synthèse.

    Le modèle déclare lui-même quels extraits fondent sa réponse
    (`cited_chunk_ids`) et s'il a pu répondre (`sufficient_context`), ce qui
    rend les citations vérifiables plutôt que présumées.
    """

    answer: str = Field(
        description="Réponse rédigée, uniquement fondée sur les extraits."
    )
    cited_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Identifiants des extraits effectivement utilisés pour répondre.",
    )
    sufficient_context: bool = Field(
        default=True,
        description="False si les extraits ne permettent pas de répondre.",
    )


class AnswerResult(BaseModel):
    """Réponse finale produite par le pipeline, avec ses citations source."""

    query: str
    answer: str
    citations: list[Citation]
    sufficient_context: bool = True


class AgentState(BaseModel):
    """État transmis de nœud en nœud le long du pipeline d'orchestration."""

    query: str
    route: RouteDecision | None = None
    retrieved: list[QueryResult] = []
    answer: AnswerResult | None = None
