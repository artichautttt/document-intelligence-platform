"""Point d'entrée de l'API FastAPI de la plateforme."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from document_intelligence.api.exception_handlers import register_exception_handlers
from document_intelligence.api.routers import documents, query
from document_intelligence.core.config import settings
from document_intelligence.core.logging import configure_logging, get_logger
from document_intelligence.orchestration.anthropic_client import AnthropicLLMClient
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.vectorization.chroma_store import ChromaStore
from document_intelligence.vectorization.qdrant_store import QdrantStore
from document_intelligence.vectorization.store import VectorStore

logger = get_logger(__name__)


def _build_store() -> VectorStore:
    if settings.vector_store_backend == "qdrant":
        return QdrantStore(url=settings.qdrant_url, collection_name=settings.qdrant_collection)
    return ChromaStore(persist_directory=settings.chroma_persist_directory)


def _build_llm(override: LLMClient | None) -> LLMClient | None:
    if override is not None:
        return override
    if settings.anthropic_api_key:
        return AnthropicLLMClient(api_key=settings.anthropic_api_key, model=settings.anthropic_model)
    # Aucun fournisseur LLM n'est configuré : `/query` répondra 503 (cf. get_llm).
    return None


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    app.state.store = _build_store()
    app.state.llm = _build_llm(app.state.llm_override)
    logger.info("api.startup", vector_store_backend=settings.vector_store_backend)
    yield
    logger.info("api.shutdown")


def create_app(llm: LLMClient | None = None) -> FastAPI:
    """Construit l'application FastAPI.

    Args:
        llm: implémentation `LLMClient` à utiliser pour `/query`. Permet
            l'injection d'un client réel (ou d'une doublure de test) sans
            modifier le code de l'application.
    """
    app = FastAPI(
        title="Document Intelligence Platform",
        description="API RAG Multi-Agents pour l'analyse de documents d'entreprise.",
        lifespan=_lifespan,
    )
    app.state.llm_override = llm

    register_exception_handlers(app)
    app.include_router(documents.router)
    app.include_router(query.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


# Point d'entrée conventionnel pour `uvicorn document_intelligence.api.app:app`.
app = create_app()
