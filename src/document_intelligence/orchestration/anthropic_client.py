"""Implémentation `LLMClient` basée sur l'API Anthropic (Claude)."""

import time
from contextlib import AbstractContextManager
from typing import Any

import anthropic

from document_intelligence.core.logging import get_logger
from document_intelligence.observability import registry, trace_span
from document_intelligence.orchestration.exceptions import LLMGenerationError
from document_intelligence.orchestration.llm import (
    LLMClient,
    T,
    validate_tool_input,
)

logger = get_logger(__name__)

_DEFAULT_MAX_TOKENS = 1024
_STRUCTURED_TOOL_NAME = "submit_answer"


def _token_count(value: Any) -> int:
    return value if isinstance(value, int) else 0


class AnthropicLLMClient(LLMClient):
    """Client LLM de production, appelant l'API Messages d'Anthropic."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def _span(self, mode: str) -> AbstractContextManager[dict[str, Any]]:
        # Attributs `gen_ai.*` : conventions OpenTelemetry GenAI, reconnues par Langfuse.
        return trace_span(
            "llm.call",
            **{
                "gen_ai.system": "anthropic",
                "gen_ai.request.model": self._model,
                "llm.mode": mode,
            },
        )

    def _record_usage(
        self, response: Any, started_at: float, span: dict[str, Any]
    ) -> None:
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

    def complete(self, prompt: str) -> str:
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
            self._record_usage(response, started_at, span)

        text_blocks = [block.text for block in response.content if block.type == "text"]
        if not text_blocks:
            raise LLMGenerationError(
                "L'API Anthropic n'a retourné aucun contenu textuel."
            )

        return "".join(text_blocks)

    def complete_structured(
        self, prompt: str, schema: type[T], system: str | None = None
    ) -> T:
        """Sortie structurée par function calling : l'outil est forcé (`tool_choice`).

        Le schéma JSON de `schema` est fourni comme `input_schema` de l'outil ;
        les arguments de l'appel d'outil sont ensuite validés par Pydantic.
        """
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
            self._record_usage(response, started_at, span)

        for block in response.content:
            if block.type == "tool_use" and block.name == _STRUCTURED_TOOL_NAME:
                return validate_tool_input(schema, block.input)

        raise LLMGenerationError("L'API Anthropic n'a retourné aucun appel d'outil.")
