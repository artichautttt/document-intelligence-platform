"""Endpoint de recherche/synthèse citée sur les documents déjà vectorisés."""

from fastapi import APIRouter, Depends

from document_intelligence.api.dependencies import (
    get_llm,
    get_store,
    get_verifier,
    verify_api_key,
)
from document_intelligence.api.schemas import QueryRequest, QueryResponse
from document_intelligence.core.config import settings
from document_intelligence.observability import trace_span
from document_intelligence.observability.prometheus import observe_query
from document_intelligence.orchestration.graph import answer_query
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.orchestration.verification import AnswerVerifier
from document_intelligence.vectorization.store import VectorStore

router = APIRouter(
    prefix="/query", tags=["query"], dependencies=[Depends(verify_api_key)]
)


@router.post("", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    store: VectorStore = Depends(get_store),
    llm: LLMClient = Depends(get_llm),
    verifier: AnswerVerifier | None = Depends(get_verifier),
) -> QueryResponse:
    """Exécute le pipeline routage → retrieval → synthèse pour la requête soumise.

    Latence, issue (répondue, refusée, erreur) et coût LLM de la requête sont
    exposés à Prometheus (`GET /metrics/prometheus`).
    """
    with observe_query() as observation, trace_span("api.query", k=request.k):
        result = answer_query(
            request.query,
            store=store,
            llm=llm,
            k=request.k,
            verifier=verifier,
            max_regenerations=settings.verification_max_regenerations,
        )
        observation.refused = not result.sufficient_context
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        citations=result.citations,
        sufficient_context=result.sufficient_context,
        verified=result.verified,
        regenerations=result.regenerations,
    )
