"""Point d'entrée public du pipeline d'orchestration multi-agents."""

from document_intelligence.core.logging import get_logger
from document_intelligence.orchestration.exceptions import OrchestrationError
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.orchestration.models import AgentState, AnswerResult
from document_intelligence.orchestration.nodes import retrieve, route_query, synthesize
from document_intelligence.vectorization.store import VectorStore

logger = get_logger(__name__)


def answer_query(query: str, store: VectorStore, llm: LLMClient, k: int = 5) -> AnswerResult:
    """Exécute le pipeline routage → retrieval → synthèse pour une requête utilisateur.

    Args:
        query: question posée par l'utilisateur.
        store: vector store interrogé pour la recherche de contexte.
        llm: client LLM utilisé pour générer la réponse.
        k: nombre de chunks récupérés pour la synthèse.
    """
    state = AgentState(query=query)
    state = route_query(state)
    state = retrieve(state, store=store, k=k)
    state = synthesize(state, llm=llm)

    if state.answer is None:
        raise OrchestrationError("Le nœud de synthèse n'a produit aucune réponse.")

    logger.info(
        "orchestration.success",
        query=query,
        chunk_count=len(state.answer.citations),
    )
    return state.answer
