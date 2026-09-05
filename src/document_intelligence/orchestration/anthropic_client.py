"""Implémentation `LLMClient` basée sur l'API Anthropic (Claude)."""

import anthropic

from document_intelligence.core.logging import get_logger
from document_intelligence.orchestration.exceptions import LLMGenerationError
from document_intelligence.orchestration.llm import LLMClient

logger = get_logger(__name__)

_DEFAULT_MAX_TOKENS = 1024


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

    def complete(self, prompt: str) -> str:
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

        return "".join(text_blocks)
