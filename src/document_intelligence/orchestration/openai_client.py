"""Implémentation `LLMClient` basée sur l'API Chat Completions d'OpenAI."""

from typing import Any

import httpx
import openai

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
        token_count(getattr(usage, "prompt_tokens", 0)),
        token_count(getattr(usage, "completion_tokens", 0)),
    )


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system}] if system else []
    return [*messages, {"role": "user", "content": prompt}]


class OpenAILLMClient(InstrumentedLLMClient):
    """Client LLM appelant l'API Chat Completions d'OpenAI.

    Les sorties structurées passent par un appel de fonction forcé
    (`tool_choice` nommé), comme pour Anthropic, plutôt que par
    `response_format` strict : le mode strict exige que tous les champs
    soient requis, ce que les schémas à valeurs par défaut ne respectent pas.

    Args:
        http_client: client HTTP injectable (tests : transport simulé).
    """

    provider = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        *,
        role: str = "default",
        cache: LLMResponseCache | None = None,
        ledger: CostLedger | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        super().__init__(model, max_tokens, role=role, cache=cache, ledger=ledger)
        self._client = openai.OpenAI(api_key=api_key, http_client=http_client)

    def _create(self, **kwargs: Any) -> Any:
        try:
            return self._client.chat.completions.create(
                model=self._model, max_completion_tokens=self._max_tokens, **kwargs
            )
        except openai.OpenAIError as exc:
            raise LLMGenerationError(
                f"L'appel à l'API OpenAI a échoué : {exc}"
            ) from exc

    def _raw_complete(self, prompt: str) -> RawCompletion:
        response = self._create(messages=_messages(prompt, None))
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LLMGenerationError("L'API OpenAI n'a retourné aucun contenu textuel.")
        return RawCompletion(content, *_usage(response))

    def _raw_structured(
        self, prompt: str, schema: type[T], system: str | None
    ) -> RawCompletion:
        response = self._create(
            messages=_messages(prompt, system),
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": STRUCTURED_TOOL_NAME,
                        "description": STRUCTURED_TOOL_DESCRIPTION,
                        "parameters": schema.model_json_schema(),
                    },
                }
            ],
            tool_choice={
                "type": "function",
                "function": {"name": STRUCTURED_TOOL_NAME},
            },
        )
        message = response.choices[0].message if response.choices else None
        for call in (message.tool_calls if message else None) or []:
            if call.function.name == STRUCTURED_TOOL_NAME:
                return RawCompletion(call.function.arguments, *_usage(response))
        raise LLMGenerationError("L'API OpenAI n'a retourné aucun appel de fonction.")
