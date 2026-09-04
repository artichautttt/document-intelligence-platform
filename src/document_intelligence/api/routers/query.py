"""Endpoint de recherche/synthèse citée sur les documents déjà vectorisés."""

from fastapi import APIRouter, Depends

from document_intelligence.api.dependencies import get_llm, get_store
from document_intelligence.api.schemas import QueryRequest, QueryResponse
from document_intelligence.orchestration.graph import answer_query
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.vectorization.store import VectorStore

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    store: VectorStore = Depends(get_store),
    llm: LLMClient = Depends(get_llm),
) -> QueryResponse:
    """Exécute le pipeline routage → retrieval → synthèse pour la requête soumise."""
    result = answer_query(request.query, store=store, llm=llm, k=request.k)
    return QueryResponse(query=result.query, answer=result.answer, citations=result.citations)
