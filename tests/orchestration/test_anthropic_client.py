"""Tests de l'implémentation Anthropic de LLMClient (API mockée, aucun appel réseau)."""

from unittest.mock import MagicMock, patch

import anthropic
import pytest

from document_intelligence.orchestration.anthropic_client import AnthropicLLMClient
from document_intelligence.orchestration.exceptions import LLMGenerationError


def _text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


class TestAnthropicLLMClient:
    def test_complete_returns_text_from_response(self) -> None:
        client = AnthropicLLMClient(api_key="fake-key")
        with patch.object(
            client._client.messages, "create", return_value=_text_response("Reponse generee")
        ) as mock_create:
            result = client.complete("Quelle est la question ?")

        assert result == "Reponse generee"
        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["messages"] == [
            {"role": "user", "content": "Quelle est la question ?"}
        ]

    def test_complete_raises_llm_generation_error_on_api_error(self) -> None:
        client = AnthropicLLMClient(api_key="fake-key")
        api_error = anthropic.APIError(
            message="boom", request=MagicMock(), body=None
        )
        with patch.object(client._client.messages, "create", side_effect=api_error):
            with pytest.raises(LLMGenerationError):
                client.complete("prompt")

    def test_complete_raises_when_response_has_no_text_block(self) -> None:
        client = AnthropicLLMClient(api_key="fake-key")
        empty_response = MagicMock()
        empty_response.content = []
        with patch.object(client._client.messages, "create", return_value=empty_response):
            with pytest.raises(LLMGenerationError):
                client.complete("prompt")
