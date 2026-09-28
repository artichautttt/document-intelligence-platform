"""Model tiering : choix du modèle Claude selon le rôle de l'appel LLM.

- `SYNTHESIS` : réponse finale du RAG — `LLM_SYNTHESIS_MODEL`, sinon `ANTHROPIC_MODEL`.
- `JUDGE` : LLM-juge de l'évaluation — `LLM_JUDGE_MODEL`, sinon `ANTHROPIC_MODEL`.
- `LIGHT` : étapes simples (reformulation, classification, extraction de
  métadonnées) — `LLM_LIGHT_MODEL`, Claude Haiku par défaut.

`ANTHROPIC_MODEL` reste le modèle de référence : sans surcharge, le
comportement est identique à celui d'avant le tiering.
"""

from enum import Enum
from pathlib import Path

from document_intelligence.core.config import Settings, settings
from document_intelligence.orchestration.anthropic_client import AnthropicLLMClient
from document_intelligence.orchestration.cache import LLMResponseCache
from document_intelligence.orchestration.cost import CostLedger


class LLMRole(str, Enum):
    SYNTHESIS = "synthesis"
    JUDGE = "judge"
    LIGHT = "light"


def model_for(role: LLMRole, config: Settings = settings) -> str:
    """Modèle configuré pour `role`."""
    if role is LLMRole.SYNTHESIS:
        return config.llm_synthesis_model or config.anthropic_model
    if role is LLMRole.JUDGE:
        return config.llm_judge_model or config.anthropic_model
    return config.llm_light_model


def build_response_cache(config: Settings = settings) -> LLMResponseCache | None:
    """Cache des réponses LLM selon la configuration ; `None` s'il est désactivé."""
    if not config.llm_cache_enabled:
        return None
    return LLMResponseCache(
        ttl_seconds=config.llm_cache_ttl_seconds,
        max_entries=config.llm_cache_max_entries,
        path=Path(config.llm_cache_path) if config.llm_cache_path else None,
    )


def build_llm_client(
    role: LLMRole,
    api_key: str,
    *,
    cache: LLMResponseCache | None = None,
    ledger: CostLedger | None = None,
    config: Settings = settings,
) -> AnthropicLLMClient:
    """Client Anthropic utilisant le modèle du tier associé à `role`."""
    return AnthropicLLMClient(
        api_key=api_key,
        model=model_for(role, config),
        role=role.value,
        cache=cache,
        ledger=ledger,
    )
