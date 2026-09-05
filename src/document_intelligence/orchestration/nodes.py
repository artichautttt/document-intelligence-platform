"""Nœuds du pipeline d'agents : routage, retrieval, synthèse, citation.

Chaque nœud est une fonction pure `AgentState -> AgentState`, conçue pour
correspondre 1:1 aux nœuds d'un `StateGraph` LangGraph : l'introduction de
la dépendance LangGraph (prévue mais non ajoutée à ce sprint, cf. README)
consistera à enregistrer ces mêmes fonctions comme nœuds du graphe sans
modifier leur logique.
"""

from document_intelligence.orchestration.exceptions import (
    EmptyQueryError,
    NoRelevantChunkError,
)
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.orchestration.models import (
    AgentState,
    AnswerResult,
    Citation,
    RouteDecision,
)
from document_intelligence.vectorization.store import VectorStore

_SYNTHESIS_PROMPT_TEMPLATE = (
    "Réponds à la question suivante en te basant uniquement sur les extraits fournis.\n\n"
    "Question : {query}\n\n"
    "Extraits :\n{context}"
)


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

    context = "\n\n".join(
        f"[{result.chunk_id}] {result.text}" for result in state.retrieved
    )
    prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(query=state.query, context=context)
    answer_text = llm.complete(prompt)

    citations = [
        Citation(
            chunk_id=result.chunk_id, provenance=result.provenance, score=result.score
        )
        for result in state.retrieved
    ]

    answer = AnswerResult(query=state.query, answer=answer_text, citations=citations)
    return state.model_copy(update={"answer": answer})
