"""Nœuds du pipeline d'agents : routage, retrieval, synthèse, citation.

Chaque nœud est une fonction pure `AgentState -> AgentState`, conçue pour
correspondre 1:1 aux nœuds d'un `StateGraph` LangGraph : l'introduction de
la dépendance LangGraph (prévue mais non ajoutée à ce sprint, cf. README)
consistera à enregistrer ces mêmes fonctions comme nœuds du graphe sans
modifier leur logique.
"""

from document_intelligence.core.logging import get_logger
from document_intelligence.orchestration.exceptions import (
    EmptyQueryError,
    NoRelevantChunkError,
)
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.orchestration.models import (
    AgentState,
    AnswerResult,
    Citation,
    GroundedAnswer,
    RouteDecision,
)
from document_intelligence.orchestration.prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    build_synthesis_prompt,
)
from document_intelligence.vectorization.store import VectorStore

logger = get_logger(__name__)


def route_query(state: AgentState) -> AgentState:
    """Détermine la stratégie de traitement d'une requête (routage minimal du Sprint 3).

    Raises:
        EmptyQueryError: si la requête est vide ou ne contient que des espaces.
    """
    if not state.query.strip():
        raise EmptyQueryError("La requête utilisateur est vide.")

    return state.model_copy(update={"route": RouteDecision(route="retrieval_qa")})


def retrieve(state: AgentState, store: VectorStore, k: int = 5) -> AgentState:
    """Récupère les chunks les plus pertinents pour la requête via le vector store.

    Raises:
        NoRelevantChunkError: si aucun chunk n'est retrouvé pour la requête.
    """
    results = store.query(state.query, k=k)
    if not results:
        raise NoRelevantChunkError(
            f"Aucun chunk pertinent trouvé pour la requête : {state.query!r}"
        )

    return state.model_copy(update={"retrieved": results})


def synthesize(state: AgentState, llm: LLMClient) -> AgentState:
    """Génère une réponse citée à partir des chunks récupérés.

    Raises:
        NoRelevantChunkError: si aucun chunk n'a été récupéré au préalable.
    """
    if not state.retrieved:
        raise NoRelevantChunkError(
            "Impossible de synthétiser une réponse sans chunk récupéré."
        )

    prompt = build_synthesis_prompt(state.query, state.retrieved)
    grounded = llm.complete_structured(
        prompt, GroundedAnswer, system=SYNTHESIS_SYSTEM_PROMPT
    )

    # Seules les citations déclarées par le modèle ET présentes dans le contexte
    # récupéré sont conservées ; un identifiant inconnu est une hallucination.
    cited_ids = set(grounded.cited_chunk_ids)
    known_ids = {result.chunk_id for result in state.retrieved}
    unknown_ids = cited_ids - known_ids
    if unknown_ids:
        logger.warning(
            "synthesis.unknown_citations",
            chunk_ids=sorted(unknown_ids),
            query=state.query,
        )
    if grounded.sufficient_context and not cited_ids & known_ids:
        logger.warning("synthesis.no_valid_citation", query=state.query)

    citations = [
        Citation(
            chunk_id=result.chunk_id, provenance=result.provenance, score=result.score
        )
        for result in state.retrieved
        if result.chunk_id in cited_ids and grounded.sufficient_context
    ]

    answer = AnswerResult(
        query=state.query,
        answer=grounded.answer,
        citations=citations,
        sufficient_context=grounded.sufficient_context,
    )
    return state.model_copy(update={"answer": answer})
