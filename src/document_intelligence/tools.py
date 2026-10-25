"""Outils exposés aux clients externes : serveur MCP et agent à tool calling.

Chaque outil est une fonction mince au-dessus des interfaces existantes
(`VectorStore`, backend RAG configuré) : aucune logique de recherche ou de
synthèse n'est dupliquée ici. Les entrées sont validées et les sorties sont des modèles
Pydantic sérialisables (sortie structurée MCP, résultats d'outil Anthropic).
"""

from pydantic import BaseModel, Field

from document_intelligence.orchestration.backends import RAGBackend
from document_intelligence.orchestration.exceptions import (
    EmptyQueryError,
    OrchestrationError,
)
from document_intelligence.vectorization.models import DocumentSummary
from document_intelligence.vectorization.store import VectorStore

MAX_K = 20


class ToolUnavailableError(OrchestrationError):
    """Levée quand un outil dépend d'un service non configuré (ex. aucun LLM)."""


class SearchHit(BaseModel):
    """Un chunk retrouvé par `search_documents`."""

    chunk_id: str
    document_id: str
    source_path: str
    score: float
    text: str


class SearchResult(BaseModel):
    query: str
    hits: list[SearchHit]


class Source(BaseModel):
    """Chunk cité par la réponse de `ask_question`."""

    chunk_id: str
    document_id: str
    source_path: str
    score: float


class CitedAnswer(BaseModel):
    """Réponse du pipeline RAG avec ses sources."""

    question: str
    answer: str
    sufficient_context: bool = Field(
        description="False si les documents ne permettent pas de répondre (refus)."
    )
    verified: bool | None = None
    sources: list[Source]


class DocumentList(BaseModel):
    documents: list[DocumentSummary]


def _check_query(text: str) -> str:
    if not text.strip():
        raise EmptyQueryError("La requête est vide.")
    return text


def _check_k(k: int) -> int:
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k doit être compris entre 1 et {MAX_K} (reçu : {k}).")
    return k


def search_documents(store: VectorStore, query: str, k: int = 5) -> SearchResult:
    """Recherche les `k` chunks les plus pertinents, sans appel LLM."""
    results = store.query(_check_query(query), k=_check_k(k))
    return SearchResult(
        query=query,
        hits=[
            SearchHit(
                chunk_id=r.chunk_id,
                document_id=r.provenance.document_id,
                source_path=r.provenance.source_path,
                score=r.score,
                text=r.text,
            )
            for r in results
        ],
    )


def ask_question(backend: RAGBackend | None, question: str, k: int = 5) -> CitedAnswer:
    """Répond à une question via le backend RAG configuré, avec les sources citées.

    Raises:
        ToolUnavailableError: si aucun fournisseur LLM n'est configuré.
    """
    if backend is None:
        raise ToolUnavailableError(
            "Aucun fournisseur LLM n'est configuré (clé d'API absente) : "
            "ask_question est indisponible, search_documents reste utilisable."
        )
    result = backend.answer(_check_query(question), k=_check_k(k))
    return CitedAnswer(
        question=result.query,
        answer=result.answer,
        sufficient_context=result.sufficient_context,
        verified=result.verified,
        sources=[
            Source(
                chunk_id=c.chunk_id,
                document_id=c.provenance.document_id,
                source_path=c.provenance.source_path,
                score=c.score,
            )
            for c in result.citations
        ],
    )


def list_documents(store: VectorStore) -> DocumentList:
    """Documents indexés, avec leur nombre de chunks."""
    return DocumentList(documents=store.list_documents())
