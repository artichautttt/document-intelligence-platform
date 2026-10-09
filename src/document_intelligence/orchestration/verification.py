"""Vérification des réponses générées : contrat `AnswerVerifier` et prompts.

Le vérificateur reçoit la question, la réponse et les extraits sources, et
décide si chaque affirmation est soutenue (`VerificationVerdict.grounded`).
Le graphe (`orchestration/graph.py`) régénère la réponse avec le retour du
vérificateur si elle ne l'est pas, dans la limite de `max_regenerations`.

L'implémentation fournie est un agent CrewAI (`crew_verifier.CrewAIVerifier`) ;
le contrat permet d'en substituer une autre (doublure de test, autre framework).
"""

from abc import ABC, abstractmethod
from html import escape

from document_intelligence.orchestration.models import VerificationVerdict
from document_intelligence.vectorization.models import QueryResult


class AnswerVerifier(ABC):
    """Contrat d'un agent de vérification de réponse."""

    @abstractmethod
    def verify(
        self, query: str, answer: str, sources: list[QueryResult]
    ) -> VerificationVerdict:
        """Vérifie que `answer` est entièrement soutenue par `sources`.

        Raises:
            LLMOutputError: si le verdict produit est inexploitable.
            LLMGenerationError: si l'appel au modèle échoue.
        """


def format_sources(sources: list[QueryResult]) -> str:
    """Extraits balisés, échappés comme dans le prompt de synthèse."""
    return "\n".join(
        f'<extrait id="{escape(s.chunk_id)}">{escape(s.text, quote=False)}</extrait>'
        for s in sources
    )


def build_feedback_block(feedback: str) -> str:
    """Bloc ajouté au prompt de synthèse lors d'une régénération."""
    return (
        "\n<retour_verification>\n"
        "Une version précédente de ta réponse contenait des affirmations non "
        "soutenues par les extraits. Corrige-la : n'affirme que ce qui figure "
        "explicitement dans les extraits, ou mets `sufficient_context` à false.\n"
        f"{escape(feedback, quote=False)}\n"
        "</retour_verification>"
    )


def describe_unsupported(verdict: VerificationVerdict) -> str:
    """Retour textuel d'un verdict négatif, transmis au rédacteur."""
    claims = "\n".join(f"- {claim}" for claim in verdict.unsupported_claims)
    parts = [verdict.feedback.strip()] if verdict.feedback.strip() else []
    if claims:
        parts.append(f"Affirmations non soutenues :\n{claims}")
    return "\n".join(parts) or "Réponse jugée non fondée sur les extraits."
