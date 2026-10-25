"""Construction des services partagés (vector store, LLM, vérificateur) depuis la config.

Utilisé par l'API FastAPI (`api.app`) et par le serveur MCP (`mcp_server`), pour
que les deux points d'entrée servent exactement le même pipeline.
"""

from document_intelligence.core.config import settings
from document_intelligence.orchestration.backends import (
    LangGraphBackend,
    RAGBackend,
    check_backend,
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


def build_store() -> VectorStore:
    """Vector store désigné par `VECTOR_STORE_BACKEND` et `RETRIEVAL_MODE`."""
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


def build_llm(override: LLMClient | None = None) -> LLMClient | None:
    """Client LLM de synthèse ; `None` si aucune clé de fournisseur n'est configurée."""
    if override is not None:
        return override
    if provider_api_key():
        # Fournisseur choisi par LLM_PROVIDER (anthropic, openai ou mistral).
        return build_llm_client(LLMRole.SYNTHESIS, cache=build_response_cache())
    return None


def build_verifier(
    override: AnswerVerifier | None, llm: LLMClient | None
) -> AnswerVerifier | None:
    """Agent de vérification désigné par `ANSWER_VERIFICATION` ; `None` si désactivé."""
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


def build_backend(
    store: VectorStore,
    llm: LLMClient | None,
    verifier: AnswerVerifier | None = None,
) -> RAGBackend | None:
    """Backend de réponse désigné par `RAG_BACKEND` ; `None` sans fournisseur LLM.

    `llm` (éventuellement une doublure injectée) sert le backend `langgraph` ;
    le backend `langchain` construit son propre modèle de chat Anthropic. Avec
    `GUARDRAILS_ENABLED`, le backend est enveloppé dans `GuardedBackend`.
    """
    backend = _build_raw_backend(store, llm, verifier)
    if backend is None or not settings.guardrails_enabled:
        return backend
    from document_intelligence.guardrails import GuardedBackend

    return GuardedBackend(backend, min_overlap=settings.guardrails_min_overlap)


def _build_raw_backend(
    store: VectorStore,
    llm: LLMClient | None,
    verifier: AnswerVerifier | None,
) -> RAGBackend | None:

    if check_backend(settings.rag_backend) == "langchain":
        if not settings.anthropic_api_key:
            return None
        # Import différé : groupe de dépendances optionnel `langchain`.
        from document_intelligence.orchestration.langchain_backend import (
            build_langchain_rag,
        )

        return build_langchain_rag(store)
    if llm is None:
        return None
    return LangGraphBackend(
        store,
        llm,
        verifier=verifier,
        max_regenerations=settings.verification_max_regenerations,
    )
