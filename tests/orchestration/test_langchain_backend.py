"""Backend LangChain (chaîne LCEL) avec un modèle de chat factice, sans réseau."""

import importlib.util
import json
import re
from collections.abc import Sequence
from typing import Any

import anthropic
import httpx
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from document_intelligence.orchestration.cost import CostLedger
from document_intelligence.orchestration.exceptions import (
    EmptyQueryError,
    LLMGenerationError,
    LLMOutputError,
    NoRelevantChunkError,
)
from document_intelligence.orchestration.langchain_backend import (
    LangChainRAG,
    VectorStoreRetriever,
    documents_to_results,
)
from document_intelligence.orchestration.prompts import SYNTHESIS_SYSTEM_PROMPT

from ..mcp_server.conftest import InMemoryStore, demo_store

MODEL = "claude-sonnet-5"


@pytest.fixture
def store() -> InMemoryStore:
    return demo_store()


class ScriptedChatModel(BaseChatModel):
    """Modèle de chat factice ; chaque appel consomme une étape du script.

    - "cite" : appel d'outil valide citant le premier extrait du prompt ;
    - "stringified" : `cited_chunk_ids` sérialisé en chaîne JSON (à réparer) ;
    - "no_tool" : réponse texte sans appel d'outil ;
    - "api_error" : lève `anthropic.APIError`.
    """

    script: list[str]
    prompts: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append(messages)
        step = self.script.pop(0)
        if step == "api_error":
            raise anthropic.APIError(
                "surcharge", httpx.Request("POST", "https://api"), body=None
            )
        ids = re.findall(r'<extrait id="([^"]+)"', str(messages[-1].content))
        usage = {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050}
        if step == "no_tool":
            message = AIMessage(content="Je ne sais pas.", usage_metadata=usage)
        else:
            cited: Any = ids[:1] if step == "cite" else json.dumps(ids[:1])
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "GroundedAnswer",
                        "args": {
                            "answer": "La redevance est de 184 000 euros HT.",
                            "cited_chunk_ids": cited,
                            "sufficient_context": True,
                        },
                        "id": "toolu_1",
                        "type": "tool_call",
                    }
                ],
                usage_metadata=usage,
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def _rag(
    store: InMemoryStore, *script: str, ledger: CostLedger | None = None
) -> tuple[LangChainRAG, ScriptedChatModel]:
    model = ScriptedChatModel(script=list(script), prompts=[])
    return LangChainRAG(store, model, MODEL, ledger), model


def test_retriever_keeps_provenance_round_trip(store: InMemoryStore) -> None:
    documents = VectorStoreRetriever(store=store, k=2).invoke("redevance")

    assert [d.metadata["source_path"] for d in documents] == ["/docs/contrat.pdf"] * 2
    results = documents_to_results(documents)
    assert [r.chunk_id for r in results] == [c.chunk_id for c in store.chunks[:2]]
    assert results[0].provenance == store.chunks[0].provenance
    assert results[0].score == 0.9


def test_answer_cites_retrieved_chunk_and_records_cost(store: InMemoryStore) -> None:
    ledger = CostLedger()
    rag, model = _rag(store, "cite", ledger=ledger)

    result = rag.answer("Quel est le montant de la redevance ?", k=2)

    assert store.queries == [("Quel est le montant de la redevance ?", 2)]
    assert result.answer == "La redevance est de 184 000 euros HT."
    assert [c.chunk_id for c in result.citations] == [store.chunks[0].chunk_id]
    assert result.citations[0].provenance == store.chunks[0].provenance
    # mêmes prompts que le graphe LangGraph
    system, human = model.prompts[0]
    assert system.content == SYNTHESIS_SYSTEM_PROMPT
    assert "<question>" in str(human.content)
    (record,) = ledger.records
    assert (record.role, record.model) == ("synthesis", MODEL)
    assert (record.input_tokens, record.output_tokens) == (1000, 50)


def test_stringified_list_is_repaired(store: InMemoryStore) -> None:
    rag, _ = _rag(store, "stringified")

    result = rag.answer("Montant ?", k=1)

    assert [c.chunk_id for c in result.citations] == [store.chunks[0].chunk_id]


def test_invalid_output_is_retried_once(store: InMemoryStore) -> None:
    ledger = CostLedger()
    rag, _ = _rag(store, "no_tool", "cite", ledger=ledger)

    result = rag.answer("Montant ?", k=1)

    assert result.sufficient_context is True
    assert len(ledger.records) == 2  # la tentative ratée est aussi facturée


def test_invalid_output_twice_raises(store: InMemoryStore) -> None:
    rag, _ = _rag(store, "no_tool", "no_tool")

    with pytest.raises(LLMOutputError):
        rag.answer("Montant ?", k=1)


def test_api_error_becomes_generation_error(store: InMemoryStore) -> None:
    rag, _ = _rag(store, "api_error")

    with pytest.raises(LLMGenerationError, match="surcharge"):
        rag.answer("Montant ?", k=1)


def test_empty_query_raises_before_any_call(store: InMemoryStore) -> None:
    rag, model = _rag(store)

    with pytest.raises(EmptyQueryError):
        rag.answer("   ")
    assert model.prompts == []


def test_no_chunk_raises_without_calling_the_model() -> None:
    rag, model = _rag(InMemoryStore([]), "cite")

    with pytest.raises(NoRelevantChunkError):
        rag.answer("Montant ?")
    assert model.prompts == []


@pytest.mark.skipif(
    importlib.util.find_spec("langchain_anthropic") is None,
    reason="groupe optionnel langchain non installé",
)
def test_build_chat_model_configures_chat_anthropic() -> None:
    from document_intelligence.orchestration.instrumented import DEFAULT_MAX_TOKENS
    from document_intelligence.orchestration.langchain_backend import build_chat_model

    chat = build_chat_model("cle-de-test", MODEL)

    assert chat.model == MODEL
    assert chat.max_tokens == DEFAULT_MAX_TOKENS
