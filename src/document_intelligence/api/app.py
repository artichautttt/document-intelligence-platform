"""Point d'entrée de l'API FastAPI de la plateforme."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from document_intelligence.api.exception_handlers import register_exception_handlers
from document_intelligence.api.routers import documents, metrics, query
from document_intelligence.core.config import settings
from document_intelligence.core.logging import configure_logging, get_logger
from document_intelligence.observability import (
    SqliteMetricsStore,
    registry,
    remove_sink,
)
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.orchestration.verification import AnswerVerifier
from document_intelligence.services import (
    build_backend,
    build_llm,
    build_store,
    build_verifier,
)

if TYPE_CHECKING:
    from document_intelligence.observability.otel import OtelSink

logger = get_logger(__name__)


def _start_otel() -> "OtelSink | None":
    if not settings.otel_exporter_otlp_endpoint:
        return None
    try:
        from document_intelligence.observability.otel import setup_otel
    except ImportError:
        logger.warning(
            "otel.unavailable",
            hint="installer le groupe de dépendances `otel` (uv sync --group otel)",
        )
        return None
    return setup_otel(
        endpoint=settings.otel_exporter_otlp_endpoint,
        headers=settings.otel_exporter_otlp_headers,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    app.state.store = build_store()
    # Sans fournisseur LLM configuré, `app.state.llm` vaut None : `/query` répond 503.
    app.state.llm = build_llm(app.state.llm_override)
    app.state.verifier = build_verifier(app.state.verifier_override, app.state.llm)
    # Backend de réponse (RAG_BACKEND) ; None sans fournisseur LLM -> 503.
    app.state.backend = build_backend(
        app.state.store, app.state.llm, app.state.verifier
    )

    metrics_store: SqliteMetricsStore | None = None
    if settings.metrics_db_path:
        metrics_store = SqliteMetricsStore(settings.metrics_db_path)
        registry.attach_store(metrics_store)
    otel_sink = _start_otel()

    logger.info(
        "api.startup",
        vector_store_backend=settings.vector_store_backend,
        retrieval_mode=settings.retrieval_mode,
        llm_provider=settings.llm_provider,
        rag_backend=settings.rag_backend,
        answer_verification=app.state.verifier is not None,
        metrics_persistent=metrics_store is not None,
        otel_enabled=otel_sink is not None,
    )
    yield
    if otel_sink is not None:
        remove_sink(otel_sink)
        otel_sink.shutdown()
    if metrics_store is not None:
        registry.detach_store()
        metrics_store.close()
    logger.info("api.shutdown")


def create_app(
    llm: LLMClient | None = None, verifier: AnswerVerifier | None = None
) -> FastAPI:
    """Construit l'application FastAPI.

    Args:
        llm: implémentation `LLMClient` à utiliser pour `/query`. Permet
            l'injection d'un client réel (ou d'une doublure de test) sans
            modifier le code de l'application.
        verifier: agent de vérification à utiliser ; par défaut, celui que
            désigne `ANSWER_VERIFICATION`.
    """
    app = FastAPI(
        title="Document Intelligence Platform",
        description="API RAG Multi-Agents pour l'analyse de documents d'entreprise.",
        lifespan=_lifespan,
    )
    app.state.llm_override = llm
    app.state.verifier_override = verifier

    register_exception_handlers(app)
    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(metrics.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


# Point d'entrée conventionnel pour `uvicorn document_intelligence.api.app:app`.
app = create_app()
