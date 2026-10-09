"""Socle commun des clients LLM concrets : cache, coût, tracing et nouvelle tentative.

Chaque fournisseur (Anthropic, OpenAI, Mistral) n'implémente que deux appels
bruts — complétion texte et appel d'outil forcé — et hérite de tout le reste,
afin que le tiering, le cache, le journal de coût et les métriques se
comportent à l'identique quel que soit le fournisseur actif.
"""

import json
import time
from abc import abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from document_intelligence.core.logging import get_logger
from document_intelligence.observability import registry, trace_span
from document_intelligence.observability.prometheus import record_llm_cost
from document_intelligence.orchestration.cache import LLMResponseCache, cache_key
from document_intelligence.orchestration.cost import (
    CostLedger,
    LLMCallRecord,
    cost_usd,
)
from document_intelligence.orchestration.exceptions import LLMOutputError
from document_intelligence.orchestration.llm import (
    LLMClient,
    T,
    validate_tool_input,
)

logger = get_logger(__name__)

DEFAULT_MAX_TOKENS = 1024
STRUCTURED_TOOL_NAME = "submit_answer"
STRUCTURED_TOOL_DESCRIPTION = "Soumet la réponse structurée finale."
_OUTPUT_RETRIES = 1


def token_count(value: Any) -> int:
    return value if isinstance(value, int) else 0


@dataclass(frozen=True)
class RawCompletion:
    """Résultat brut d'un appel fournisseur : contenu et tokens consommés.

    `content` est le texte (mode `complete`) ou les arguments de l'appel
    d'outil, éventuellement encore sérialisés en JSON (mode `structured`).
    """

    content: Any
    input_tokens: int
    output_tokens: int


class InstrumentedLLMClient(LLMClient):
    """Client LLM instrumenté, indépendant du fournisseur.

    Args:
        model: identifiant du modèle chez le fournisseur.
        role: rôle de l'appel (synthèse, juge...), reporté dans le coût et le tracing.
        cache: cache exact-match consulté avant tout appel API.
        ledger: journal de coût alimenté à chaque appel (API ou cache).
    """

    #: Valeur de l'attribut `gen_ai.system` (conventions OpenTelemetry GenAI).
    provider: str = "unknown"

    def __init__(
        self,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        *,
        role: str = "default",
        cache: LLMResponseCache | None = None,
        ledger: CostLedger | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._role = role
        self._cache = cache
        self._ledger = ledger

    @property
    def model(self) -> str:
        return self._model

    @abstractmethod
    def _raw_complete(self, prompt: str) -> RawCompletion:
        """Appel texte libre au fournisseur.

        Raises:
            LLMGenerationError: erreur d'API ou réponse sans contenu textuel.
        """

    @abstractmethod
    def _raw_structured(
        self, prompt: str, schema: type[T], system: str | None
    ) -> RawCompletion:
        """Appel d'outil forcé dont le schéma d'entrée est `schema`.

        Raises:
            LLMGenerationError: erreur d'API ou réponse sans appel d'outil.
        """

    def _span(self, mode: str) -> AbstractContextManager[dict[str, Any]]:
        # Attributs `gen_ai.*` : conventions OpenTelemetry GenAI, reconnues par Langfuse.
        return trace_span(
            "llm.call",
            **{
                "gen_ai.system": self.provider,
                "gen_ai.request.model": self._model,
                "llm.mode": mode,
                "llm.role": self._role,
            },
        )

    def _call(self, mode: str, call: Any) -> RawCompletion:
        """Exécute un appel brut dans un span et comptabilise ses tokens."""
        started_at = time.perf_counter()
        with self._span(mode) as span:
            raw: RawCompletion = call()
            span["gen_ai.usage.input_tokens"] = raw.input_tokens
            span["gen_ai.usage.output_tokens"] = raw.output_tokens
            registry.record_llm_call(
                input_tokens=raw.input_tokens,
                output_tokens=raw.output_tokens,
                latency_ms=(time.perf_counter() - started_at) * 1000,
            )
        self._record_cost(raw.input_tokens, raw.output_tokens, cached=False)
        return raw

    def _record_cost(self, input_tokens: int, output_tokens: int, cached: bool) -> None:
        if self._ledger is not None:
            self._ledger.record(
                LLMCallRecord(
                    role=self._role,
                    model=self._model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached=cached,
                )
            )
        record_llm_cost(
            provider=self.provider,
            model=self._model,
            role=self._role,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0
            if cached
            else cost_usd(self._model, input_tokens, output_tokens),
            cached=cached,
        )

    def _cached(self, key: str) -> Any | None:
        """Valeur en cache pour `key` ; un hit est tracé et comptabilisé sans appel API."""
        if self._cache is None:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        with trace_span(
            "llm.cache_hit",
            **{"gen_ai.request.model": self._model, "llm.role": self._role},
        ):
            self._record_cost(entry.input_tokens, entry.output_tokens, cached=True)
        return entry.value

    def _store(self, key: str, value: Any, raw: RawCompletion) -> None:
        if self._cache is not None:
            self._cache.set(
                key,
                value,
                input_tokens=raw.input_tokens,
                output_tokens=raw.output_tokens,
            )

    def complete(self, prompt: str) -> str:
        key = cache_key("complete", self._model, prompt)
        cached = self._cached(key)
        if isinstance(cached, str):
            return cached

        raw = self._call("complete", lambda: self._raw_complete(prompt))
        text = str(raw.content)
        self._store(key, text, raw)
        return text

    def complete_structured(
        self, prompt: str, schema: type[T], system: str | None = None
    ) -> T:
        """Sortie structurée par function calling (appel d'outil forcé).

        Une sortie invalide (observée avec le vrai modèle : balisage d'appel
        d'outil écrit dans un champ texte, champ requis absent) est retentée
        `_OUTPUT_RETRIES` fois. Les erreurs d'API ne sont jamais retentées ici.
        Seules les sorties valides sont mises en cache.

        Raises:
            LLMOutputError: si la sortie reste invalide après les nouvelles tentatives.
            LLMGenerationError: si l'appel à l'API échoue.
        """
        key = cache_key(
            "structured",
            self._model,
            system,
            prompt,
            json.dumps(schema.model_json_schema(), sort_keys=True),
        )
        cached = self._cached(key)
        if cached is not None:
            return schema.model_validate(cached)

        for attempt in range(_OUTPUT_RETRIES + 1):
            raw = self._call(
                "structured", lambda: self._raw_structured(prompt, schema, system)
            )
            try:
                result = validate_tool_input(schema, _decode_arguments(raw.content))
            except LLMOutputError:
                if attempt == _OUTPUT_RETRIES:
                    raise
                logger.warning(
                    "llm.structured_retry", schema=schema.__name__, attempt=attempt + 1
                )
                continue
            self._store(key, result.model_dump(mode="json"), raw)
            return result
        raise AssertionError("inatteignable")  # pragma: no cover


def _decode_arguments(arguments: Any) -> Any:
    """Arguments d'appel d'outil : OpenAI et Mistral les renvoient en chaîne JSON."""
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(
            f"Les arguments de l'appel d'outil ne sont pas du JSON valide : {exc}"
        ) from exc
