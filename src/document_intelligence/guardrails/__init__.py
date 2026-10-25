"""Garde-fous : règles d'injection (entrée, ingestion) et vérificateur de sortie."""

from document_intelligence.guardrails.backend import GuardedBackend, guard_output
from document_intelligence.guardrails.ingestion import ScreeningResult, screen_chunks
from document_intelligence.guardrails.output import OutputVerdict, verify_output
from document_intelligence.guardrails.rules import RULES, RuleMatch, detect

__all__ = [
    "RULES",
    "GuardedBackend",
    "OutputVerdict",
    "RuleMatch",
    "ScreeningResult",
    "detect",
    "guard_output",
    "screen_chunks",
    "verify_output",
]
