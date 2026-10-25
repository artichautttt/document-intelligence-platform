"""Nœuds du pipeline d'agents : routage, retrieval, synthèse, citation, vérification.

Chaque nœud est une fonction pure `AgentState -> AgentState`, conçue pour
correspondre 1:1 aux nœuds d'un `StateGraph` LangGraph : l'introduction de
la dépendance LangGraph (prévue mais non ajoutée à ce sprint, cf. README)
consistera à enregistrer ces mêmes fonctions comme nœuds du graphe sans
modifier leur logique.
"""

from document_intelligence.core.logging import get_logger
from document_intelligence.observability.prometheus import record_verification
from document_intelligence.orchestration.exceptions import (
    EmptyQueryError,
    LLMOutputError,
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
from document_intelligence.orchestration.verification import (
    AnswerVerifier,
    build_feedback_block,
    describe_unsupported,
)
from document_intelligence.vectorization.models import QueryResult
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


def build_answer(
    query: str, retrieved: list[QueryResult], grounded: GroundedAnswer
) -> AnswerResult:
    """Transforme la sortie structurée du modèle en réponse citée.

    Partagée par tous les backends de synthèse (graphe LangGraph, chaîne
    LangChain) : seules les citations déclarées par le modèle ET présentes dans
    le contexte récupéré sont conservées ; un identifiant inconnu est une
    hallucination.
    """
    cited_ids = set(grounded.cited_chunk_ids)
    known_ids = {result.chunk_id for result in retrieved}
    unknown_ids = cited_ids - known_ids
    if unknown_ids:
        logger.warning(
            "synthesis.unknown_citations",
            chunk_ids=sorted(unknown_ids),
            query=query,
        )
    if grounded.sufficient_context and not cited_ids & known_ids:
        logger.warning("synthesis.no_valid_citation", query=query)

    citations = [
        Citation(
            chunk_id=result.chunk_id, provenance=result.provenance, score=result.score
        )
        for result in retrieved
        if result.chunk_id in cited_ids and grounded.sufficient_context
    ]
    return AnswerResult(
        query=query,
        answer=grounded.answer,
        citations=citations,
        sufficient_context=grounded.sufficient_context,
    )


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
    regenerations = state.regenerations
    rejected = list(state.answer.rejected_answers) if state.answer else []
    if state.verification_feedback is not None:
        if state.answer is not None:
            rejected.append(state.answer.answer)
        # Régénération : le prompt diffère, donc le cache ne resert jamais
        # la réponse rejetée par le vérificateur.
        prompt += build_feedback_block(state.verification_feedback)
        regenerations += 1
    grounded = llm.complete_structured(
        prompt, GroundedAnswer, system=SYNTHESIS_SYSTEM_PROMPT
    )
    answer = build_answer(state.query, state.retrieved, grounded).model_copy(
        update={
            "verifications": state.verifications,
            "regenerations": regenerations,
            "rejected_answers": rejected,
        }
    )
    return state.model_copy(
        update={
            "answer": answer,
            "regenerations": regenerations,
            "verification_feedback": None,
        }
    )


def verify_answer(state: AgentState, verifier: AnswerVerifier) -> AgentState:
    """Fait vérifier la réponse par l'agent ; prépare une régénération si besoin.

    Une abstention n'est pas vérifiée (aucune affirmation à contrôler). Un
    verdict inexploitable laisse la réponse non vérifiée (`verified=None`)
    sans interrompre le pipeline ; une erreur d'API est propagée.
    """
    answer = state.answer
    if answer is None or not answer.sufficient_context:
        return state

    try:
        verdict = verifier.verify(state.query, answer.answer, state.retrieved)
    except LLMOutputError:
        logger.warning("verification.invalid_verdict", query=state.query, exc_info=True)
        return state

    record_verification(verdict.grounded)
    if not verdict.grounded:
        logger.warning(
            "verification.flagged",
            query=state.query,
            unsupported_claims=verdict.unsupported_claims,
            regenerations=state.regenerations,
        )
    verifications = [*state.verifications, verdict]
    return state.model_copy(
        update={
            "answer": answer.model_copy(
                update={"verifications": verifications, "verified": verdict.grounded}
            ),
            "verifications": verifications,
            "verification_feedback": (
                None if verdict.grounded else describe_unsupported(verdict)
            ),
        }
    )
