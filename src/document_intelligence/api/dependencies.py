"""Dépendances FastAPI donnant accès aux services applicatifs partagés.

Le vector store et le client LLM sont instanciés une fois au démarrage de
l'application (voir `api.app.create_app`) et exposés via `app.state`, afin
d'être injectables et substituables par des doublures dans les tests.
"""

from fastapi import Header, HTTPException, Request, status

from document_intelligence.core.config import settings
from document_intelligence.orchestration.backends import RAGBackend
from document_intelligence.vectorization.store import VectorStore


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Vérifie la clé d'API transmise dans l'en-tête `X-API-Key`.

    Si `settings.api_key` n'est pas configuré, l'authentification est
    désactivée (mode développement) : c'est le comportement historique de
    l'API, conservé pour ne pas casser les déploiements existants qui n'ont
    pas encore défini de clé.

    Raises:
        HTTPException: 401 si la clé est absente ou ne correspond pas.
    """
    # Chaîne vide = non configurée : `docker-compose.yml` transmet `API_KEY=`
    # quand la variable n'est pas définie sur l'hôte.
    if not settings.api_key:
        return

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé d'API absente ou invalide.",
        )


def get_store(request: Request) -> VectorStore:
    """Retourne le vector store partagé de l'application."""
    return request.app.state.store  # type: ignore[no-any-return]


def get_backend(request: Request) -> RAGBackend:
    """Retourne le backend de réponse RAG partagé (`RAG_BACKEND`).

    Raises:
        HTTPException: 503 si aucun fournisseur LLM n'est configuré.
    """
    backend: RAGBackend | None = request.app.state.backend
    if backend is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Aucun fournisseur LLM n'est configuré sur ce déploiement.",
        )
    return backend
