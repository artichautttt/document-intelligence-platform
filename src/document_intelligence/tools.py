"""Outils exposés aux clients externes : serveur MCP et agent à tool calling.

Chaque outil est une fonction mince au-dessus des interfaces existantes
(`VectorStore`, backend RAG configuré, exécuteur SQL protégé, agent de données) :
aucune logique de recherche, de synthèse ou d'exécution n'est dupliquée ici. Les entrées sont validées et les sorties sont des modèles
Pydantic sérialisables (sortie structurée MCP, résultats d'outil Anthropic).
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from document_intelligence.data_agent.schema import SCHEMA, DatabaseSchema
from document_intelligence.data_agent.sql_guard import SafeSQLExecutor, SQLResult
from document_intelligence.orchestration.backends import RAGBackend
from document_intelligence.orchestration.exceptions import (
    EmptyQueryError,
    OrchestrationError,
)
from document_intelligence.vectorization.models import DocumentSummary
from document_intelligence.vectorization.store import VectorStore

if TYPE_CHECKING:
    from document_intelligence.orchestration.data_agent import DataAgent

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


class DataQueryAnswer(BaseModel):
    """Réponse de l'agent de données, avec la requête exécutée qui la fonde."""

    question: str
    answer: str = Field(description="Réponse, suivie de la requête SQL exécutée.")
    value: float | str | None
    refused: bool = Field(description="True si la base ne permet pas de répondre.")
    sql: str | None = Field(description="Requête exécutée qui fonde la réponse.")
    blocked_queries: int = Field(
        description="Requêtes du modèle refusées par les garde-fous SQL."
    )


def describe_tables() -> DatabaseSchema:
    """Schéma documenté de la base de l'agent de données."""
    return SCHEMA


def run_sql(executor: SafeSQLExecutor, sql: str) -> SQLResult:
    """Exécute une requête SELECT derrière les garde-fous (sans appel LLM).

    Raises:
        SQLGuardError: requête refusée (écriture, table non autorisée, délai...).
        SQLExecutionError: requête autorisée mais invalide pour SQLite.
    """
    return executor.execute(sql)


def ask_data(agent: "DataAgent | None", question: str) -> DataQueryAnswer:
    """Question en langage naturel → SQL exécuté → réponse citant la requête.

    Raises:
        ToolUnavailableError: si aucune clé Anthropic n'est configurée.
    """
    if agent is None:
        raise ToolUnavailableError(
            "Aucune clé Anthropic n'est configurée : ask_data est indisponible, "
            "run_sql et describe_tables restent utilisables."
        )
    result = agent.run(_check_query(question))
    return DataQueryAnswer(
        question=question,
        answer=result.formatted(),
        value=result.value,
        refused=result.refused,
        sql=result.cited_sql,
        blocked_queries=len(result.blocked_calls),
    )
