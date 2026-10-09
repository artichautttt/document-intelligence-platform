"""Choix du fournisseur LLM (`LLM_PROVIDER`) et model tiering par rôle d'appel.

Fournisseur actif : `LLM_PROVIDER` = `anthropic` (défaut), `openai` ou
`mistral`. Son modèle de référence est respectivement `ANTHROPIC_MODEL`,
`OPENAI_MODEL` ou `MISTRAL_MODEL` ; c'est aussi la référence « sans
optimisation » du calcul d'économies.

Rôles :
- `SYNTHESIS` : réponse finale du RAG — `LLM_SYNTHESIS_MODEL`, sinon la référence.
- `JUDGE` : LLM-juge de l'évaluation — `LLM_JUDGE_MODEL`, sinon la référence.
- `VERIFIER` : agent de vérification des réponses — `LLM_VERIFIER_MODEL`,
  sinon la référence.
- `LIGHT` : étapes simples (reformulation, classification, extraction de
  métadonnées) — `LLM_LIGHT_MODEL` (Claude Haiku par défaut) avec Anthropic,
  la référence avec les autres fournisseurs.

Sans surcharge et avec `LLM_PROVIDER=anthropic`, le comportement est
identique à celui d'avant l'ajout des fournisseurs.
"""

from enum import Enum
from pathlib import Path

from document_intelligence.core.config import Settings, settings
from document_intelligence.orchestration.anthropic_client import AnthropicLLMClient
from document_intelligence.orchestration.cache import LLMResponseCache
from document_intelligence.orchestration.cost import CostLedger
from document_intelligence.orchestration.instrumented import InstrumentedLLMClient
from document_intelligence.orchestration.mistral_client import MistralLLMClient
from document_intelligence.orchestration.openai_client import OpenAILLMClient

PROVIDERS = ("anthropic", "openai", "mistral")


class LLMRole(str, Enum):
    SYNTHESIS = "synthesis"
    JUDGE = "judge"
    VERIFIER = "verifier"
    LIGHT = "light"


def active_provider(config: Settings = settings) -> str:
    """Fournisseur configuré, normalisé.

    Raises:
        ValueError: si `LLM_PROVIDER` ne désigne pas un fournisseur supporté.
    """
    provider = config.llm_provider.strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER={config.llm_provider!r} non supporté ; "
            f"valeurs possibles : {', '.join(PROVIDERS)}."
        )
    return provider


def reference_model(config: Settings = settings) -> str:
    """Modèle de référence du fournisseur actif."""
    return {
        "anthropic": config.anthropic_model,
        "openai": config.openai_model,
        "mistral": config.mistral_model,
    }[active_provider(config)]


def provider_api_key(config: Settings = settings) -> str | None:
    """Clé d'API du fournisseur actif (`None` si non configurée)."""
    return {
        "anthropic": config.anthropic_api_key,
        "openai": config.openai_api_key,
        "mistral": config.mistral_api_key,
    }[active_provider(config)]


def model_for(role: LLMRole, config: Settings = settings) -> str:
    """Modèle configuré pour `role` chez le fournisseur actif."""
    if role is LLMRole.SYNTHESIS:
        return config.llm_synthesis_model or reference_model(config)
    if role is LLMRole.JUDGE:
        return config.llm_judge_model or reference_model(config)
    if role is LLMRole.VERIFIER:
        return config.llm_verifier_model or reference_model(config)
    if active_provider(config) == "anthropic":
        return config.llm_light_model
    return reference_model(config)


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
    api_key: str | None = None,
    *,
    cache: LLMResponseCache | None = None,
    ledger: CostLedger | None = None,
    config: Settings = settings,
) -> InstrumentedLLMClient:
    """Client du fournisseur actif, avec le modèle du tier associé à `role`.

    Args:
        api_key: clé explicite ; par défaut celle du fournisseur actif.

    Raises:
        ValueError: fournisseur inconnu ou clé d'API absente.
    """
    provider = active_provider(config)
    key = api_key or provider_api_key(config)
    if not key:
        raise ValueError(f"Aucune clé d'API configurée pour le fournisseur {provider}.")
    model = model_for(role, config)
    client_class: (
        type[AnthropicLLMClient] | type[OpenAILLMClient] | type[MistralLLMClient]
    )
    if provider == "openai":
        client_class = OpenAILLMClient
    elif provider == "mistral":
        client_class = MistralLLMClient
    else:
        client_class = AnthropicLLMClient
    return client_class(key, model, role=role.value, cache=cache, ledger=ledger)
