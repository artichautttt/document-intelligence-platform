"""Dépendances FastAPI donnant accès aux services applicatifs partagés.

Le vector store et le client LLM sont instanciés une fois au démarrage de
l'application (voir `api.app.create_app`) et exposés via `app.state`, afin
d'être injectables et substituables par des doublures dans les tests.
"""

from fastapi import HTTPException, Request, status

from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.vectorization.store import VectorStore


def get_store(request: Request) -> VectorStore:
    """Retourne le vector store partagé de l'application."""
    return request.app.state.store  # type: ignore[no-any-return]


def get_llm(request: Request) -> LLMClient:
    """Retourne le client LLM partagé de l'application.

    Raises:
        HTTPException: 503 si aucun `LLMClient` n'a été configuré (aucune
            implémentation concrète n'est encore fournie par la plateforme).
    """
    llm: LLMClient | None = request.app.state.llm
    if llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Aucun fournisseur LLM n'est configuré sur ce déploiement.",
        )
    return llm
