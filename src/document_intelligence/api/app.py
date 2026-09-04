"""Point d'entrée de l'API FastAPI de la plateforme."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from document_intelligence.api.exception_handlers import register_exception_handlers
from document_intelligence.api.routers import documents, query
from document_intelligence.core.config import settings
from document_intelligence.core.logging import configure_logging, get_logger
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.vectorization.chroma_store import ChromaStore

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    app.state.store = ChromaStore(persist_directory=settings.chroma_persist_directory)
    # Aucune implémentation concrète de `LLMClient` n'est encore branchée (cf. README) :
    # `/query` répondra 503 tant qu'un fournisseur n'est pas passé à `create_app`.
    app.state.llm = app.state.llm_override
    logger.info("api.startup")
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
