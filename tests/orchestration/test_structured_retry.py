"""Nouvelle tentative sur sortie structurée invalide (cas observé avec le vrai modèle)."""

from unittest.mock import MagicMock, patch

import anthropic
import pytest
from pydantic import BaseModel

from document_intelligence.orchestration.anthropic_client import AnthropicLLMClient
from document_intelligence.orchestration.exceptions import (
    LLMGenerationError,
    LLMOutputError,
)


class Verdict(BaseModel):
    score: int
    reasoning: str


def _tool_response(payload: dict[str, object]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "submit_answer"
    block.input = payload
    response = MagicMock()
    response.content = [block]
    return response


# Forme observée : le balisage d'appel d'outil fuit dans `reasoning`, `score` manque.
_LEAKED = {"reasoning": 'ok</reasoning>\n<parameter name="score">5'}
_VALID = {"score": 5, "reasoning": "ok"}


def test_invalid_output_is_retried_once_and_can_succeed() -> None:
    client = AnthropicLLMClient(api_key="fake-key")
    with patch.object(
        client._client.messages,
        "create",
        side_effect=[_tool_response(_LEAKED), _tool_response(_VALID)],
    ) as create:
        result = client.complete_structured("prompt", Verdict)

    assert result == Verdict(score=5, reasoning="ok")
    assert create.call_count == 2


def test_output_still_invalid_after_retry_raises() -> None:
    client = AnthropicLLMClient(api_key="fake-key")
    with (
        patch.object(
            client._client.messages,
            "create",
            side_effect=[_tool_response(_LEAKED), _tool_response(_LEAKED)],
        ) as create,
        pytest.raises(LLMOutputError),
    ):
        client.complete_structured("prompt", Verdict)

    assert create.call_count == 2  # une seule nouvelle tentative, pas de boucle


def test_api_errors_are_never_retried() -> None:
    client = AnthropicLLMClient(api_key="fake-key")
    api_error = anthropic.APIError(message="boom", request=MagicMock(), body=None)
    with (
        patch.object(
            client._client.messages, "create", side_effect=api_error
        ) as create,
        pytest.raises(LLMGenerationError),
    ):
        client.complete_structured("prompt", Verdict)

    assert create.call_count == 1


def test_valid_output_costs_a_single_call() -> None:
    client = AnthropicLLMClient(api_key="fake-key")
    with patch.object(
        client._client.messages, "create", return_value=_tool_response(_VALID)
    ) as create:
        client.complete_structured("prompt", Verdict)

    assert create.call_count == 1
