"""Implémentation `LLMClient` basée sur l'API Chat Completion de Mistral AI."""

from typing import Any

from mistralai.client import Mistral
from mistralai.client.errors import MistralError, NoResponseError
from mistralai.client.httpclient import HttpClient

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


def _text(content: Any) -> str:
    """Contenu d'un message : chaîne, ou liste de fragments (`TextChunk`...)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(getattr(chunk, "text", "") or "" for chunk in content)
    return ""


class MistralLLMClient(InstrumentedLLMClient):
    """Client LLM appelant l'API Chat Completion de Mistral AI.

    Les sorties structurées passent par un appel de fonction forcé
    (`tool_choice` nommé), comme pour Anthropic et OpenAI.

    Args:
        http_client: client HTTP injectable (tests : transport simulé).
    """

    provider = "mistral"

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-small-latest",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        *,
        role: str = "default",
        cache: LLMResponseCache | None = None,
        ledger: CostLedger | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        super().__init__(model, max_tokens, role=role, cache=cache, ledger=ledger)
        self._client = Mistral(api_key=api_key, client=http_client)

    def _create(self, **kwargs: Any) -> Any:
        try:
            return self._client.chat.complete(
                model=self._model, max_tokens=self._max_tokens, **kwargs
            )
        except (MistralError, NoResponseError) as exc:
            raise LLMGenerationError(
                f"L'appel à l'API Mistral a échoué : {exc}"
            ) from exc

    def _raw_complete(self, prompt: str) -> RawCompletion:
        response = self._create(messages=_messages(prompt, None))
        text = _text(response.choices[0].message.content) if response.choices else ""
        if not text:
            raise LLMGenerationError(
                "L'API Mistral n'a retourné aucun contenu textuel."
            )
        return RawCompletion(text, *_usage(response))

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
        raise LLMGenerationError("L'API Mistral n'a retourné aucun appel de fonction.")
