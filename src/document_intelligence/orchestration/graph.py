"""Point d'entrée public du pipeline d'orchestration multi-agents.

Le pipeline est construit comme un `StateGraph` LangGraph dont les nœuds sont
directement les fonctions pures `AgentState -> AgentState` de
`orchestration/nodes.py` (routage → retrieval → synthèse), conformément au
plan annoncé au Sprint 3 : introduire LangGraph sans changer la logique
métier des nœuds.
"""

from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from document_intelligence.core.logging import get_logger
from document_intelligence.orchestration.exceptions import OrchestrationError
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.orchestration.models import AgentState, AnswerResult
from document_intelligence.orchestration.nodes import retrieve, route_query, synthesize
from document_intelligence.vectorization.store import VectorStore

logger = get_logger(__name__)


def _build_graph(store: VectorStore, llm: LLMClient, k: int) -> CompiledStateGraph[Any, Any, Any, Any]:
    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    graph.add_node("route", route_query)
    graph.add_node("retrieve", partial(retrieve, store=store, k=k))
    graph.add_node("synthesize", partial(synthesize, llm=llm))

    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def answer_query(query: str, store: VectorStore, llm: LLMClient, k: int = 5) -> AnswerResult:
    """Exécute le pipeline routage → retrieval → synthèse pour une requête utilisateur.

    Args:
        query: question posée par l'utilisateur.
        store: vector store interrogé pour la recherche de contexte.
        llm: client LLM utilisé pour générer la réponse.
        k: nombre de chunks récupérés pour la synthèse.
    """
    compiled_graph = _build_graph(store=store, llm=llm, k=k)
    final_state = compiled_graph.invoke(AgentState(query=query))
    answer = final_state["answer"] if isinstance(final_state, dict) else final_state.answer

    if not isinstance(answer, AnswerResult):
        raise OrchestrationError("Le nœud de synthèse n'a produit aucune réponse.")

    logger.info(
        "orchestration.success",
        query=query,
        chunk_count=len(answer.citations),
    )
    return answer
