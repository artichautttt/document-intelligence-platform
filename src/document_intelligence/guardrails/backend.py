"""Backend RAG gardé : règles d'injection sur la question, vérification de la sortie."""

from dataclasses import dataclass, field

from document_intelligence.core.logging import get_logger
from document_intelligence.guardrails.output import DEFAULT_MIN_OVERLAP, verify_output
from document_intelligence.guardrails.rules import detect
from document_intelligence.orchestration.backends import RAGBackend
from document_intelligence.orchestration.models import AnswerResult, GuardrailReport

logger = get_logger(__name__)

INPUT_REFUSAL = (
    "Requête refusée : elle contient des instructions adressées au système "
    "plutôt qu'une question sur les documents."
)
OUTPUT_REFUSAL = "Les documents ne permettent pas de fournir une réponse vérifiable à cette question."
ABSTENTION = "Les documents ne permettent pas de répondre à cette question."


def guard_output(
    result: AnswerResult, min_overlap: float = DEFAULT_MIN_OVERLAP
) -> AnswerResult:
    """Applique le vérificateur de sortie à une réponse du pipeline.

    Une abstention reçoit un message standard (le texte libre du modèle n'est
    pas renvoyé : il pourrait relayer une injection). Une réponse est gardée,
    dégradée ou remplacée par un refus selon le verdict de `verify_output`.
    """
    if not result.sufficient_context:
        return result.model_copy(
            update={
                "answer": ABSTENTION,
                "guardrail": GuardrailReport(action="abstention"),
            }
        )
    cited = {c.chunk_id for c in result.citations}
    texts = [r.text for r in result.context if r.chunk_id in cited]
    verdict = verify_output(result.answer, texts, min_overlap)
    removed = [c.sentence for c in verdict.removed]
    reasons = [c.reason for c in verdict.removed]
    if verdict.action == "pass":
        return result.model_copy(update={"guardrail": GuardrailReport(action="pass")})
    logger.warning(
        "guardrails.output_flagged",
        query=result.query,
        action=verdict.action,
        reasons=reasons,
    )
    if verdict.action == "degrade":
        return result.model_copy(
            update={
                "answer": verdict.answer,
                "guardrail": GuardrailReport(
                    action="degraded", removed_sentences=removed, reasons=reasons
                ),
            }
        )
    return result.model_copy(
        update={
            "answer": OUTPUT_REFUSAL,
            "citations": [],
            "sufficient_context": False,
            "guardrail": GuardrailReport(
                action="refused_output", removed_sentences=removed, reasons=reasons
            ),
        }
    )


@dataclass
class GuardedBackend:
    """Enveloppe un `RAGBackend` : question filtrée, réponse vérifiée."""

    inner: RAGBackend
    min_overlap: float = DEFAULT_MIN_OVERLAP
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"{self.inner.name}+guardrails"

    def answer(self, query: str, k: int = 5) -> AnswerResult:
        matches = detect(query, "question")
        if matches:
            rules = [m.rule for m in matches]
            logger.warning("guardrails.input_blocked", query=query, rules=rules)
            return AnswerResult(
                query=query,
                answer=INPUT_REFUSAL,
                citations=[],
                sufficient_context=False,
                guardrail=GuardrailReport(action="blocked_input", rules=rules),
            )
        return guard_output(self.inner.answer(query, k), self.min_overlap)
