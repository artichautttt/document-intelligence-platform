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
from document_intelligence.orchestration.tiering import (
    LLMRole,
    build_llm_client,
    build_response_cache,
    provider_api_key,
)
from document_intelligence.orchestration.verification import AnswerVerifier
from document_intelligence.vectorization.chroma_store import ChromaStore
from document_intelligence.vectorization.hybrid_store import HybridStore
from document_intelligence.vectorization.qdrant_store import QdrantStore
from document_intelligence.vectorization.store import VectorStore

if TYPE_CHECKING:
    from document_intelligence.observability.otel import OtelSink

logger = get_logger(__name__)


def _build_store() -> VectorStore:
    store: VectorStore
    if settings.vector_store_backend == "qdrant":
        store = QdrantStore(
            url=settings.qdrant_url, collection_name=settings.qdrant_collection
        )
    else:
        store = ChromaStore(persist_directory=settings.chroma_persist_directory)
    if settings.retrieval_mode == "hybrid":
        return HybridStore(store)
    return store


def _build_llm(override: LLMClient | None) -> LLMClient | None:
    if override is not None:
        return override
    if provider_api_key():
        # Fournisseur choisi par LLM_PROVIDER (anthropic, openai ou mistral).
        return build_llm_client(LLMRole.SYNTHESIS, cache=build_response_cache())
    # Aucun fournisseur LLM n'est configuré : `/query` répondra 503 (cf. get_llm).
    return None


def _build_verifier(
    override: AnswerVerifier | None, llm: LLMClient | None
) -> AnswerVerifier | None:
    if override is not None:
        return override
    if settings.answer_verification == "none" or llm is None:
        return None
    if settings.answer_verification != "crewai":
        raise ValueError(
            f"ANSWER_VERIFICATION={settings.answer_verification!r} non supporté "
            "(valeurs : none, crewai)."
        )
    # Import différé : CrewAI est optionnel (groupe `agents`) et lent à importer.
    from document_intelligence.orchestration.crew_verifier import CrewAIVerifier

    return CrewAIVerifier(
        build_llm_client(LLMRole.VERIFIER, cache=build_response_cache())
    )


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
    app.state.store = _build_store()
    app.state.llm = _build_llm(app.state.llm_override)
    app.state.verifier = _build_verifier(app.state.verifier_override, app.state.llm)

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
