"""Agent de recherche (tool calling natif) avec un client Anthropic scripté."""

import copy
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from document_intelligence.orchestration.backends import LangGraphBackend
from document_intelligence.orchestration.cost import CostLedger, cost_usd
from document_intelligence.orchestration.exceptions import LLMGenerationError
from document_intelligence.orchestration.research_agent import (
    TOOL_DEFINITIONS,
    ResearchAgent,
)

from ..mcp_server.conftest import CitingLLM, InMemoryStore, demo_store

MODEL = "claude-sonnet-5"


@pytest.fixture
def store() -> InMemoryStore:
    return demo_store()


def tool_use(block_id: str, name: str, **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=arguments)


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def response(*blocks: SimpleNamespace, stop: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=500, output_tokens=40),
    )


class ScriptedClient:
    """Imite `anthropic.Anthropic` : `messages.create` renvoie les réponses du script."""

    def __init__(self, *responses: SimpleNamespace | Exception) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        # copie : l'agent complète ensuite la même liste de messages
        self.requests.append(copy.deepcopy(kwargs))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _tool_results(request: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = request["messages"][-1]["content"]
    return content


def test_runs_tools_chosen_by_the_model_and_returns_final_answer(
    store: InMemoryStore,
) -> None:
    client = ScriptedClient(
        response(
            text("Je cherche."),
            tool_use("t1", "search_documents", query="redevance", k=2),
        ),
        response(tool_use("t2", "ask_question", question="Montant de la redevance ?")),
        response(text("184 000 euros HT (contrat.pdf)."), stop="end_turn"),
    )
    ledger = CostLedger()
    llm = CitingLLM()
    agent = ResearchAgent(
        client, MODEL, store, LangGraphBackend(store, llm), ledger=ledger
    )

    result = agent.run("Quelle est la redevance ?")

    assert result.answer == "184 000 euros HT (contrat.pdf)."
    assert (result.stop_reason, result.steps) == ("end_turn", 3)
    assert [(c.step, c.name, c.is_error) for c in result.tool_calls] == [
        (1, "search_documents", False),
        (2, "ask_question", False),
    ]
    assert store.queries[0] == ("redevance", 2)
    assert llm.calls == 1
    # chaque résultat d'outil revient au modèle, rattaché à son appel
    (first,) = _tool_results(client.requests[1])
    assert (first["type"], first["tool_use_id"], first["is_error"]) == (
        "tool_result",
        "t1",
        False,
    )
    assert "/docs/contrat.pdf" in first["content"]
    (second,) = _tool_results(client.requests[2])
    assert "184 000 euros HT" in second["content"]
    # outils et tokens
    assert client.requests[0]["tools"] == TOOL_DEFINITIONS
    assert (result.input_tokens, result.output_tokens) == (1500, 120)
    assert result.cost_usd == pytest.approx(cost_usd(MODEL, 1500, 120))
    assert [r.role for r in ledger.records] == ["agent"] * 3


def test_tool_errors_are_returned_to_the_model(store: InMemoryStore) -> None:
    client = ScriptedClient(
        response(
            tool_use("t1", "search_documents", query="x", k=0),
            tool_use("t2", "ask_question", question="Montant ?"),
            tool_use("t3", "delete_everything"),
            tool_use("t4", "search_documents"),
        ),
        response(text("Impossible de répondre."), stop="end_turn"),
    )
    agent = ResearchAgent(client, MODEL, store, backend=None)

    result = agent.run("Montant ?")

    assert [c.is_error for c in result.tool_calls] == [True] * 4
    contents = [r["content"] for r in _tool_results(client.requests[1])]
    assert "k doit" in contents[0]
    assert "Aucun fournisseur LLM" in contents[1]
    assert "Outil inconnu" in contents[2]
    assert contents[3].startswith("KeyError")
    assert result.answer == "Impossible de répondre."


def test_stops_after_max_steps(store: InMemoryStore) -> None:
    loop = [
        response(tool_use(f"t{i}", "search_documents", query="x")) for i in range(3)
    ]
    agent = ResearchAgent(ScriptedClient(*loop), MODEL, store, None, max_steps=3)

    result = agent.run("?")

    assert (result.stop_reason, result.steps, len(result.tool_calls)) == (
        "max_steps",
        3,
        3,
    )


def test_api_error_interrupts_the_run(store: InMemoryStore) -> None:
    error = anthropic.APIError(
        "surcharge", httpx.Request("POST", "https://api"), body=None
    )
    agent = ResearchAgent(ScriptedClient(error), MODEL, store, None)

    with pytest.raises(LLMGenerationError, match="surcharge"):
        agent.run("?")


def test_empty_text_block_is_not_sent_back(store: InMemoryStore) -> None:
    # Observé en réel : bloc texte vide à côté d'un appel d'outil, refusé ensuite
    # par l'API s'il est renvoyé tel quel (400).
    client = ScriptedClient(
        response(text(""), tool_use("t1", "search_documents", query="x")),
        response(text("Fin."), stop="end_turn"),
    )

    ResearchAgent(client, MODEL, store, None).run("?")

    assistant = client.requests[1]["messages"][1]
    assert assistant["role"] == "assistant"
    assert [b["type"] for b in assistant["content"]] == ["tool_use"]
