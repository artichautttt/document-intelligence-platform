"""Schémas Pydantic des requêtes/réponses HTTP exposées par l'API."""

from pydantic import BaseModel, Field

from document_intelligence.orchestration.models import Citation


class DocumentIngestResponse(BaseModel):
    """Réponse renvoyée après ingestion, chunking et vectorisation d'un document."""

    document_id: str
    chunk_count: int


class QueryRequest(BaseModel):
    """Requête de recherche/synthèse soumise au pipeline d'orchestration."""

    query: str
    k: int = Field(default=5, gt=0, le=20)


class QueryResponse(BaseModel):
    """Réponse citée produite par le pipeline d'orchestration."""

    query: str
    answer: str
    citations: list[Citation]


class ErrorResponse(BaseModel):
    """Corps d'erreur uniforme retourné par les handlers d'exceptions."""

    detail: str
