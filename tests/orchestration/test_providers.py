"""Clients OpenAI et Mistral, sélection du fournisseur par `LLM_PROVIDER`.

**Aucun appel réseau** : les API OpenAI et Mistral sont simulées au niveau
HTTP (`MockTransport`). Le vrai SDK de chaque fournisseur construit la requête
et parse une réponse au format documenté de l'API ; ces tests vérifient donc
notre intégration avec les SDK, pas le comportement des modèles réels.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import httpx2
import pytest

from document_intelligence.chunking.service import chunk_document
from document_intelligence.core.config import Settings
from document_intelligence.ingestion.service import ingest
from document_intelligence.observability.prometheus import RAG_LLM_COST_USD_TOTAL
from document_intelligence.orchestration.anthropic_client import AnthropicLLMClient
from document_intelligence.orchestration.cost import CostLedger
from document_intelligence.orchestration.exceptions import (
    LLMGenerationError,
    LLMOutputError,
)
from document_intelligence.orchestration.graph import answer_query
from document_intelligence.orchestration.instrumented import InstrumentedLLMClient
from document_intelligence.orchestration.mistral_client import MistralLLMClient
from document_intelligence.orchestration.models import GroundedAnswer
from document_intelligence.orchestration.openai_client import OpenAILLMClient
from document_intelligence.orchestration.tiering import (
    LLMRole,
    build_llm_client,
    model_for,
    reference_model,
)
from document_intelligence.vectorization.chroma_store import ChromaStore

Handler = Callable[[dict[str, Any]], tuple[int, dict[str, Any]]]


def _chat_response(
    message: dict[str, Any], prompt_tokens: int = 120, completion_tokens: int = 30
) -> dict[str, Any]:
    """Corps de réponse Chat Completions (format commun OpenAI / Mistral)."""
    return {
        "id": "cmpl-test",
        "object": "chat.completion",
        "created": 1_760_000_000,
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _tool_call_message(arguments: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "submit_answer", "arguments": arguments},
            }
        ],
    }


def _grounded_from_request(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Simule un modèle qui cite le premier extrait reçu dans le prompt."""
    user = next(m for m in body["messages"] if m["role"] == "user")["content"]
    first_id = re.findall(r'<extrait id="([^"]+)"', user)[0]
    arguments = json.dumps(
        {
            "answer": "Réponse simulée fondée sur le premier extrait.",
            "cited_chunk_ids": [first_id],
            "sufficient_context": True,
        }
    )
    return 200, _chat_response(_tool_call_message(arguments))


class _Recorder:
    """Transport simulé qui enregistre les requêtes JSON reçues."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []

    def _respond(self, path: str, content: bytes) -> tuple[int, dict[str, Any]]:
        body = json.loads(content)
        self.paths.append(path)
        self.requests.append(body)
        return self.handler(body)

    def openai_client(self) -> httpx.Client:
        def handle(request: httpx.Request) -> httpx.Response:
            status, payload = self._respond(request.url.path, request.content)
            return httpx.Response(status, json=payload)

        return httpx.Client(transport=httpx.MockTransport(handle))

    def mistral_client(self) -> httpx2.Client:
        def handle(request: httpx2.Request) -> httpx2.Response:
            status, payload = self._respond(request.url.path, request.content)
            return httpx2.Response(status, json=payload)

        return httpx2.Client(transport=httpx2.MockTransport(handle))


def _openai(recorder: _Recorder, **kwargs: Any) -> OpenAILLMClient:
    return OpenAILLMClient(
        api_key="sk-test", http_client=recorder.openai_client(), **kwargs
    )


def _mistral(recorder: _Recorder, **kwargs: Any) -> MistralLLMClient:
    return MistralLLMClient(
        api_key="test", http_client=recorder.mistral_client(), **kwargs
    )


_FACTORIES = {"openai": _openai, "mistral": _mistral}


@pytest.fixture(params=["openai", "mistral"])
def provider(request: pytest.FixtureRequest) -> str:
    return str(request.param)


class TestChatCompletionClients:
    def test_complete_returns_text_and_sends_model(self, provider: str) -> None:
        recorder = _Recorder(
            lambda _: (
                200,
                _chat_response({"role": "assistant", "content": "Bonjour"}),
            )
        )
        client = _FACTORIES[provider](recorder, model="mon-modele")

        assert client.complete("Salut") == "Bonjour"
        assert recorder.requests[0]["model"] == "mon-modele"
        assert recorder.requests[0]["messages"] == [
            {"role": "user", "content": "Salut"}
        ]
        assert recorder.paths[0].endswith("/chat/completions")

    def test_structured_output_forces_the_tool_and_validates(
        self, provider: str
    ) -> None:
        arguments = json.dumps(
            {"answer": "42", "cited_chunk_ids": ["c1"], "sufficient_context": True}
        )
        recorder = _Recorder(
            lambda _: (200, _chat_response(_tool_call_message(arguments)))
        )
        client = _FACTORIES[provider](recorder)

        result = client.complete_structured(
            "question", GroundedAnswer, system="consignes"
        )

        assert result == GroundedAnswer(
            answer="42", cited_chunk_ids=["c1"], sufficient_context=True
        )
        body = recorder.requests[0]
        assert body["messages"][0] == {"role": "system", "content": "consignes"}
        assert body["tool_choice"] == {
            "type": "function",
            "function": {"name": "submit_answer"},
        }
        parameters = body["tools"][0]["function"]["parameters"]
        assert set(parameters["properties"]) == {
            "answer",
            "cited_chunk_ids",
            "sufficient_context",
        }

    def test_invalid_arguments_are_retried_once(self, provider: str) -> None:
        valid = json.dumps({"answer": "ok"})
        replies = iter(["pas du json", valid])
        recorder = _Recorder(
            lambda _: (200, _chat_response(_tool_call_message(next(replies))))
        )
        client = _FACTORIES[provider](recorder)

        assert client.complete_structured("q", GroundedAnswer).answer == "ok"
        assert len(recorder.requests) == 2

    def test_invalid_arguments_twice_raise_output_error(self, provider: str) -> None:
        recorder = _Recorder(
            lambda _: (200, _chat_response(_tool_call_message('{"answer": 1}')))
        )
        client = _FACTORIES[provider](recorder)

        with pytest.raises(LLMOutputError):
            client.complete_structured("q", GroundedAnswer)

    def test_api_error_becomes_generation_error(self, provider: str) -> None:
        recorder = _Recorder(lambda _: (401, {"message": "Unauthorized"}))
        client = _FACTORIES[provider](recorder)

        with pytest.raises(LLMGenerationError) as excinfo:
            client.complete("q")
        assert not isinstance(excinfo.value, LLMOutputError)

    def test_response_without_tool_call_raises(self, provider: str) -> None:
        recorder = _Recorder(
            lambda _: (
                200,
                _chat_response({"role": "assistant", "content": "texte libre"}),
            )
        )
        client = _FACTORIES[provider](recorder)

        with pytest.raises(LLMGenerationError):
            client.complete_structured("q", GroundedAnswer)

    def test_usage_feeds_the_cost_ledger_and_prometheus(self, provider: str) -> None:
        recorder = _Recorder(
            lambda _: (
                200,
                _chat_response(
                    {"role": "assistant", "content": "ok"},
                    prompt_tokens=1_000_000,
                    completion_tokens=0,
                ),
            )
        )
        ledger = CostLedger()
        model = "gpt-4.1-mini" if provider == "openai" else "mistral-small-latest"
        client = _FACTORIES[provider](recorder, model=model, ledger=ledger)
        counter = RAG_LLM_COST_USD_TOTAL.labels(provider, model, "default")
        before = counter._value.get()

        client.complete("q")

        assert ledger.records[0].input_tokens == 1_000_000
        expected = 0.40 if provider == "openai" else 0.15
        assert counter._value.get() - before == pytest.approx(expected)


class TestRagPipelineWithSimulatedProviders:
    """Pipeline RAG réel (ingestion, chunking, ChromaDB, LangGraph) ; seule
    l'API du fournisseur est simulée."""

    def test_answer_query_end_to_end(
        self, provider: str, fixtures_dir: Path, tmp_path: Path
    ) -> None:
        store = ChromaStore(persist_directory=str(tmp_path / "chroma"))
        store.add_chunks(chunk_document(ingest(fixtures_dir / "financial_report.pdf")))
        recorder = _Recorder(_grounded_from_request)
        client = _FACTORIES[provider](recorder)

        answer = answer_query(
            "Quel est le chiffre d'affaires ?", store=store, llm=client, k=3
        )

        assert answer.sufficient_context
        assert len(answer.citations) == 1
        user_prompt = recorder.requests[0]["messages"][-1]["content"]
        assert answer.citations[0].chunk_id in user_prompt


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class TestProviderSelection:
    @pytest.fixture(autouse=True)
    def _isolated_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "LLM_PROVIDER",
            "ANTHROPIC_MODEL",
            "OPENAI_MODEL",
            "MISTRAL_MODEL",
            "LLM_SYNTHESIS_MODEL",
            "LLM_JUDGE_MODEL",
            "LLM_VERIFIER_MODEL",
            "LLM_LIGHT_MODEL",
        ):
            monkeypatch.delenv(name, raising=False)

    @pytest.mark.parametrize(
        ("provider", "client_class", "model"),
        [
            ("anthropic", AnthropicLLMClient, "claude-sonnet-5"),
            ("openai", OpenAILLMClient, "gpt-4.1-mini"),
            ("mistral", MistralLLMClient, "mistral-small-latest"),
        ],
    )
    def test_llm_provider_selects_the_client(
        self,
        provider: str,
        client_class: type[InstrumentedLLMClient],
        model: str,
    ) -> None:
        config = _settings(llm_provider=provider)

        client = build_llm_client(LLMRole.SYNTHESIS, api_key="k", config=config)

        assert type(client) is client_class
        assert client.model == model == reference_model(config)

    def test_provider_is_read_from_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "Mistral")
        monkeypatch.setenv("MISTRAL_API_KEY", "cle-mistral")

        client = build_llm_client(LLMRole.SYNTHESIS, config=_settings())

        assert isinstance(client, MistralLLMClient)

    def test_role_overrides_apply_to_any_provider(self) -> None:
        config = _settings(
            llm_provider="openai",
            llm_synthesis_model="gpt-4.1-nano",
            llm_verifier_model="gpt-4.1",
        )

        assert model_for(LLMRole.SYNTHESIS, config) == "gpt-4.1-nano"
        assert model_for(LLMRole.VERIFIER, config) == "gpt-4.1"
        assert model_for(LLMRole.JUDGE, config) == "gpt-4.1-mini"
        # Le modèle léger par défaut est un modèle Claude : hors Anthropic, référence.
        assert model_for(LLMRole.LIGHT, config) == "gpt-4.1-mini"

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="LLM_PROVIDER"):
            build_llm_client(
                LLMRole.SYNTHESIS, api_key="k", config=_settings(llm_provider="x")
            )

    def test_missing_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="openai"):
            build_llm_client(LLMRole.SYNTHESIS, config=_settings(llm_provider="openai"))
