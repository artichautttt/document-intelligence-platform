"""Implémentation `LLMClient` basée sur l'API Anthropic (Claude)."""

from typing import Any

import anthropic

from document_intelligence.orchestration.cache import LLMResponseCache
from document_intelligence.orchestration.cost import CostLedger
from document_intelligence.orchestration.exceptions import LLMGenerationError
from document_intelligence.orchestration.instrumented import (
    DEFAULT_MAX_TOKENS,
    STRUCTURED_TOOL_DESCRIPTION,
    STRUCTURED_TOOL_NAME,
    InstrumentedLLMClient,
    RawCompletion,
    token_count,
)
from document_intelligence.orchestration.llm import T


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    return (
        token_count(getattr(usage, "input_tokens", 0)),
        token_count(getattr(usage, "output_tokens", 0)),
    )


class AnthropicLLMClient(InstrumentedLLMClient):
    """Client LLM appelant l'API Messages d'Anthropic.

    Les sorties structurées passent par un outil forcé (`tool_choice`) dont
    l'`input_schema` est le schéma JSON du modèle Pydantic attendu.
    """

    provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        *,
        role: str = "default",
        cache: LLMResponseCache | None = None,
        ledger: CostLedger | None = None,
    ) -> None:
        super().__init__(model, max_tokens, role=role, cache=cache, ledger=ledger)
        self._client = anthropic.Anthropic(api_key=api_key)

    def _raw_complete(self, prompt: str) -> RawCompletion:
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

        text_blocks = [block.text for block in response.content if block.type == "text"]
        if not text_blocks:
            raise LLMGenerationError(
                "L'API Anthropic n'a retourné aucun contenu textuel."
            )
        return RawCompletion("".join(text_blocks), *_usage(response))

    def _raw_structured(
        self, prompt: str, schema: type[T], system: str | None
    ) -> RawCompletion:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system or anthropic.omit,
                messages=[{"role": "user", "content": prompt}],
                tools=[
                    {
                        "name": STRUCTURED_TOOL_NAME,
                        "description": STRUCTURED_TOOL_DESCRIPTION,
                        "input_schema": schema.model_json_schema(),
                    }
                ],
                tool_choice={"type": "tool", "name": STRUCTURED_TOOL_NAME},
            )
        except anthropic.APIError as exc:
            raise LLMGenerationError(
                f"L'appel à l'API Anthropic a échoué : {exc}"
            ) from exc

        for block in response.content:
            if block.type == "tool_use" and block.name == STRUCTURED_TOOL_NAME:
                return RawCompletion(block.input, *_usage(response))
        raise LLMGenerationError("L'API Anthropic n'a retourné aucun appel d'outil.")
