"""Endpoint de recherche/synthèse citée sur les documents déjà vectorisés."""

from fastapi import APIRouter, Depends

from document_intelligence.api.dependencies import get_backend, verify_api_key
from document_intelligence.api.schemas import QueryRequest, QueryResponse
from document_intelligence.observability import trace_span
from document_intelligence.observability.prometheus import observe_query
from document_intelligence.orchestration.backends import RAGBackend

router = APIRouter(
    prefix="/query", tags=["query"], dependencies=[Depends(verify_api_key)]
)


@router.post("", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    backend: RAGBackend = Depends(get_backend),
) -> QueryResponse:
    """Exécute le backend RAG configuré (`RAG_BACKEND`) pour la requête soumise.

    Latence, issue (répondue, refusée, erreur) et coût LLM de la requête sont
    exposés à Prometheus (`GET /metrics/prometheus`).
    """
    with observe_query() as observation, trace_span("api.query", k=request.k):
        result = backend.answer(request.query, k=request.k)
        observation.refused = not result.sufficient_context
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        citations=result.citations,
        sufficient_context=result.sufficient_context,
        verified=result.verified,
        regenerations=result.regenerations,
    )
