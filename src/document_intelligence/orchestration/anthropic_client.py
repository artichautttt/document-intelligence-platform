"""Implémentation `LLMClient` basée sur l'API Anthropic (Claude)."""

import json
import time
from contextlib import AbstractContextManager
from typing import Any

import anthropic

from document_intelligence.core.logging import get_logger
from document_intelligence.observability import registry, trace_span
from document_intelligence.orchestration.cache import LLMResponseCache, cache_key
from document_intelligence.orchestration.cost import CostLedger, LLMCallRecord
from document_intelligence.orchestration.exceptions import (
    LLMGenerationError,
    LLMOutputError,
)
from document_intelligence.orchestration.llm import (
    LLMClient,
    T,
    validate_tool_input,
)

logger = get_logger(__name__)

_DEFAULT_MAX_TOKENS = 1024
_STRUCTURED_TOOL_NAME = "submit_answer"
_OUTPUT_RETRIES = 1


def _token_count(value: Any) -> int:
    return value if isinstance(value, int) else 0


class AnthropicLLMClient(LLMClient):
    """Client LLM de production, appelant l'API Messages d'Anthropic.

    Args:
        role: rôle de l'appel (synthèse, juge...), reporté dans le coût et le tracing.
        cache: cache exact-match consulté avant tout appel API.
        ledger: journal de coût alimenté à chaque appel (API ou cache).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        *,
        role: str = "default",
        cache: LLMResponseCache | None = None,
        ledger: CostLedger | None = None,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._role = role
        self._cache = cache
        self._ledger = ledger

    @property
    def model(self) -> str:
        return self._model

    def _span(self, mode: str) -> AbstractContextManager[dict[str, Any]]:
        # Attributs `gen_ai.*` : conventions OpenTelemetry GenAI, reconnues par Langfuse.
        return trace_span(
            "llm.call",
            **{
                "gen_ai.system": "anthropic",
                "gen_ai.request.model": self._model,
                "llm.mode": mode,
                "llm.role": self._role,
            },
        )

    def _record_usage(
        self, response: Any, started_at: float, span: dict[str, Any]
    ) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        input_tokens = _token_count(getattr(usage, "input_tokens", 0))
        output_tokens = _token_count(getattr(usage, "output_tokens", 0))
        span["gen_ai.usage.input_tokens"] = input_tokens
        span["gen_ai.usage.output_tokens"] = output_tokens
        registry.record_llm_call(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.perf_counter() - started_at) * 1000,
        )
        self._record_cost(input_tokens, output_tokens, cached=False)
        return input_tokens, output_tokens

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

    def _store(self, key: str, value: Any, usage: tuple[int, int]) -> None:
        if self._cache is not None:
            self._cache.set(key, value, input_tokens=usage[0], output_tokens=usage[1])

    def complete(self, prompt: str) -> str:
        key = cache_key("complete", self._model, prompt)
        cached = self._cached(key)
        if isinstance(cached, str):
            return cached

        started_at = time.perf_counter()
        with self._span("complete") as span:
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
            except anthropic.APIError as exc:
                raise LLMGenerationError(
                    f"L'appel à l'API Anthropic a échoué : {exc}"
                ) from exc
            usage = self._record_usage(response, started_at, span)

        text_blocks = [block.text for block in response.content if block.type == "text"]
        if not text_blocks:
            raise LLMGenerationError(
                "L'API Anthropic n'a retourné aucun contenu textuel."
            )

        text = "".join(text_blocks)
        self._store(key, text, usage)
        return text

    def complete_structured(
        self, prompt: str, schema: type[T], system: str | None = None
    ) -> T:
        """Sortie structurée par function calling : l'outil est forcé (`tool_choice`).

        Le schéma JSON de `schema` est fourni comme `input_schema` de l'outil ;
        les arguments de l'appel d'outil sont ensuite validés par Pydantic.

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
            try:
                result, usage = self._structured_attempt(prompt, schema, system)
            except LLMOutputError:
                if attempt == _OUTPUT_RETRIES:
                    raise
                logger.warning(
                    "llm.structured_retry", schema=schema.__name__, attempt=attempt + 1
                )
                continue
            self._store(key, result.model_dump(mode="json"), usage)
            return result
        raise AssertionError("inatteignable")  # pragma: no cover

    def _structured_attempt(
        self, prompt: str, schema: type[T], system: str | None
    ) -> tuple[T, tuple[int, int]]:
        started_at = time.perf_counter()
        with self._span("structured") as span:
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system or anthropic.omit,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[
                        {
                            "name": _STRUCTURED_TOOL_NAME,
                            "description": "Soumet la réponse structurée finale.",
                            "input_schema": schema.model_json_schema(),
                        }
                    ],
                    tool_choice={"type": "tool", "name": _STRUCTURED_TOOL_NAME},
                )
            except anthropic.APIError as exc:
                raise LLMGenerationError(
                    f"L'appel à l'API Anthropic a échoué : {exc}"
                ) from exc
            usage = self._record_usage(response, started_at, span)

        for block in response.content:
            if block.type == "tool_use" and block.name == _STRUCTURED_TOOL_NAME:
                return validate_tool_input(schema, block.input), usage

        raise LLMGenerationError("L'API Anthropic n'a retourné aucun appel d'outil.")
