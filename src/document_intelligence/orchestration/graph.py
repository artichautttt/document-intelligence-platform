"""Point d'entrée public du pipeline d'orchestration multi-agents.

Le pipeline est construit comme un `StateGraph` LangGraph dont les nœuds sont
directement les fonctions pures `AgentState -> AgentState` de
`orchestration/nodes.py` (routage → retrieval → synthèse), conformément au
plan annoncé au Sprint 3 : introduire LangGraph sans changer la logique
métier des nœuds.
"""

from collections.abc import Callable
from functools import partial
from typing import Any, Protocol

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from document_intelligence.core.logging import get_logger
from document_intelligence.observability import trace_span
from document_intelligence.orchestration.exceptions import OrchestrationError
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.orchestration.models import AgentState, AnswerResult
from document_intelligence.orchestration.nodes import (
    retrieve,
    route_query,
    synthesize,
    verify_answer,
)
from document_intelligence.orchestration.verification import AnswerVerifier
from document_intelligence.vectorization.store import VectorStore

logger = get_logger(__name__)


class _GraphNode(Protocol):
    """Signature de nœud attendue par LangGraph (paramètre positionnel `state`)."""

    def __call__(self, state: AgentState) -> AgentState: ...


def _traced(
    name: str,
    node: Callable[[AgentState], AgentState],
    describe: Callable[[AgentState], dict[str, Any]] | None = None,
) -> _GraphNode:
    """Enveloppe un nœud dans un span de tracing sans modifier sa logique."""

    def wrapper(state: AgentState) -> AgentState:
        with trace_span(f"node.{name}") as span:
            result = node(state)
            if describe is not None:
                span.update(describe(result))
            return result

    return wrapper


def _describe_retrieval(state: AgentState) -> dict[str, Any]:
    return {
        "hits": len(state.retrieved),
        "top_score": max((r.score for r in state.retrieved), default=None),
    }


def _describe_synthesis(state: AgentState) -> dict[str, Any]:
    answer = state.answer
    return {
        "citations": len(answer.citations) if answer else 0,
        "sufficient_context": answer.sufficient_context if answer else None,
    }


def _describe_verification(state: AgentState) -> dict[str, Any]:
    answer = state.answer
    return {
        "verified": answer.verified if answer else None,
        "regenerations": state.regenerations,
    }


def _after_verification(state: AgentState, max_regenerations: int) -> str:
    """Régénère tant que le vérificateur rejette la réponse, dans la limite fixée."""
    if (
        state.verification_feedback is not None
        and state.regenerations < max_regenerations
    ):
        return "synthesize"
    return END


def _build_graph(
    store: VectorStore,
    llm: LLMClient,
    k: int,
    verifier: AnswerVerifier | None = None,
    max_regenerations: int = 1,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    graph.add_node("route", _traced("route", route_query))
    graph.add_node(
        "retrieve",
        _traced("retrieve", partial(retrieve, store=store, k=k), _describe_retrieval),
    )
    graph.add_node(
        "synthesize",
        _traced("synthesize", partial(synthesize, llm=llm), _describe_synthesis),
    )

    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    if verifier is None:
        graph.add_edge("synthesize", END)
        return graph.compile()

    # synthèse → vérification → (régénération avec le retour du vérificateur | fin)
    graph.add_node(
        "verify",
        _traced(
            "verify",
            partial(verify_answer, verifier=verifier),
            _describe_verification,
        ),
    )
    graph.add_edge("synthesize", "verify")
    graph.add_conditional_edges(
        "verify",
        partial(_after_verification, max_regenerations=max_regenerations),
        ["synthesize", END],
    )
    return graph.compile()


def answer_query(
    query: str,
    store: VectorStore,
    llm: LLMClient,
    k: int = 5,
    verifier: AnswerVerifier | None = None,
    max_regenerations: int = 1,
) -> AnswerResult:
    """Exécute le pipeline routage → retrieval → synthèse (→ vérification) pour une requête.

    Args:
        query: question posée par l'utilisateur.
        store: vector store interrogé pour la recherche de contexte.
        llm: client LLM utilisé pour générer la réponse.
        k: nombre de chunks récupérés pour la synthèse.
        verifier: agent de vérification ; `None` = graphe sans vérification.
        max_regenerations: régénérations au plus après un verdict négatif. Au-delà,
            la réponse est rendue avec `verified=False`.
    """
    compiled_graph = _build_graph(
        store=store,
        llm=llm,
        k=k,
        verifier=verifier,
        max_regenerations=max_regenerations,
    )
    with trace_span("pipeline", k=k):
        final_state = compiled_graph.invoke(AgentState(query=query))
    answer = (
        final_state["answer"] if isinstance(final_state, dict) else final_state.answer
    )

    if not isinstance(answer, AnswerResult):
        raise OrchestrationError("Le nœud de synthèse n'a produit aucune réponse.")

    logger.info(
        "orchestration.success",
        query=query,
        chunk_count=len(answer.citations),
    )
    return answer
