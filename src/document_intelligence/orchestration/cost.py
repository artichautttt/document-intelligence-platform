"""Tarification par modèle et registre de coût des appels LLM.

Le `CostLedger` comptabilise chaque appel (API ou servi par le cache) avec le
modèle effectivement utilisé, pour comparer :
- le **coût réel** : tokens des appels API, au tarif du modèle appelé ;
- le **coût sans optimisation** : tous les appels (cache compris) facturés au
  tarif du modèle de référence, comme si tout passait par lui sans cache.

Limite connue : les tokens d'un appel Haiku sont comptés avec le tokenizer de
Haiku ; les valoriser au tarif de référence est une approximation.
"""

import threading
from dataclasses import dataclass

from document_intelligence.core.config import settings


@dataclass(frozen=True)
class ModelPrice:
    """Tarif en USD par million de tokens."""

    input_per_mtok: float
    output_per_mtok: float


# Tarifs publics de l'API Anthropic (USD / million de tokens).
MODEL_PRICES: dict[str, ModelPrice] = {
    "claude-opus-5": ModelPrice(5.0, 25.0),
    "claude-sonnet-5": ModelPrice(2.0, 10.0),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0),
}


def price_for(model: str) -> ModelPrice:
    """Tarif de `model` ; à défaut, les tarifs globaux configurés (0 par défaut)."""
    return MODEL_PRICES.get(
        model,
        ModelPrice(
            settings.llm_input_price_per_mtok, settings.llm_output_price_per_mtok
        ),
    )


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Coût estimé d'un appel à `model`."""
    price = price_for(model)
    return (
        input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok
    ) / 1_000_000


@dataclass(frozen=True)
class LLMCallRecord:
    """Un appel LLM comptabilisé ; `cached` = servi par le cache, sans appel API."""

    role: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool


class CostLedger:
    """Journal des appels LLM d'un run ; sûr en accès concurrent."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[LLMCallRecord] = []

    def record(self, record: LLMCallRecord) -> None:
        with self._lock:
            self._records.append(record)

    @property
    def records(self) -> list[LLMCallRecord]:
        with self._lock:
            return list(self._records)

    def summary(self, reference_model: str) -> dict[str, float]:
        """Coût réel vs sans optimisation, au total et par rôle.

        Args:
            reference_model: modèle qui aurait traité tous les appels sans tiering.
        """
        records = self.records
        summary: dict[str, float] = {
            "llm_calls": float(len(records)),
            "llm_cache_hits": float(sum(r.cached for r in records)),
        }
        for role in sorted({r.role for r in records}) + [None]:
            selected = [r for r in records if role is None or r.role == role]
            suffix = "" if role is None else f"_{role}"
            actual = sum(
                cost_usd(r.model, r.input_tokens, r.output_tokens)
                for r in selected
                if not r.cached
            )
            unoptimized = sum(
                cost_usd(reference_model, r.input_tokens, r.output_tokens)
                for r in selected
            )
            summary[f"cost_actual_usd{suffix}"] = round(actual, 6)
            summary[f"cost_unoptimized_usd{suffix}"] = round(unoptimized, 6)
            if unoptimized > 0:
                summary[f"cost_savings_pct{suffix}"] = round(
                    100 * (1 - actual / unoptimized), 2
                )
        return summary
