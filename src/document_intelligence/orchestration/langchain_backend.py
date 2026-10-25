"""Backend RAG alternatif construit avec LangChain (LCEL).

- `VectorStoreRetriever` : `BaseRetriever` LangChain adossé à n'importe quel
  `VectorStore` du projet (Chroma, Qdrant, hybride), provenance conservée dans
  les métadonnées des `Document`.
- `LangChainRAG` : chaîne LCEL retriever → prompt → `ChatAnthropic` en sortie
  structurée (`GroundedAnswer`, par appel d'outil forcé).

Pour que la comparaison avec le graphe LangGraph ne mesure que le framework,
la chaîne réutilise les mêmes prompts (`orchestration.prompts`), la même
réparation des arguments d'outil (`validate_tool_input`), la même nouvelle
tentative unique sur sortie invalide et la même conversion en réponse citée
(`nodes.build_answer`). Ses tokens sont versés dans le même `CostLedger`.
Différence assumée : pas de cache de réponses ni d'agent de vérification.

Dépendance optionnelle : groupe `langchain` (`langchain-anthropic`).
"""

from operator import itemgetter
from typing import TYPE_CHECKING, Any

import anthropic
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pydantic import ConfigDict, SecretStr

from document_intelligence.chunking.models import ChunkProvenance
from document_intelligence.core.config import settings
from document_intelligence.core.logging import get_logger
from document_intelligence.observability import trace_span
from document_intelligence.orchestration.cost import CostLedger, LLMCallRecord
from document_intelligence.orchestration.exceptions import (
    EmptyQueryError,
    LLMGenerationError,
    LLMOutputError,
    NoRelevantChunkError,
)
from document_intelligence.orchestration.instrumented import DEFAULT_MAX_TOKENS
from document_intelligence.orchestration.llm import validate_tool_input
from document_intelligence.orchestration.models import AnswerResult, GroundedAnswer
from document_intelligence.orchestration.nodes import build_answer
from document_intelligence.orchestration.prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    build_synthesis_prompt,
)
from document_intelligence.orchestration.tiering import LLMRole, model_for
from document_intelligence.vectorization.models import QueryResult
from document_intelligence.vectorization.store import VectorStore

if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic

logger = get_logger(__name__)

_OUTPUT_RETRIES = 1


class VectorStoreRetriever(BaseRetriever):
    """Retriever LangChain au-dessus d'un `VectorStore` du projet."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: VectorStore
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [
            Document(
                id=result.chunk_id,
                page_content=result.text,
                metadata={
                    "chunk_id": result.chunk_id,
                    "score": result.score,
                    **result.provenance.model_dump(),
                },
            )
            for result in self.store.query(query, k=self.k)
        ]


def documents_to_results(documents: list[Document]) -> list[QueryResult]:
    """`Document` LangChain → `QueryResult` du projet (provenance comprise)."""
    return [
        QueryResult(
            chunk_id=str(doc.metadata["chunk_id"]),
            text=doc.page_content,
            score=float(doc.metadata["score"]),
            provenance=ChunkProvenance(
                document_id=str(doc.metadata["document_id"]),
                source_path=str(doc.metadata["source_path"]),
                element_ids=list(doc.metadata["element_ids"]),
            ),
        )
        for doc in documents
    ]


def _messages(inputs: dict[str, Any]) -> list[BaseMessage]:
    results: list[QueryResult] = inputs["results"]
    if not results:
        raise NoRelevantChunkError(
            f"Aucun chunk pertinent trouvé pour la requête : {inputs['query']!r}"
        )
    return [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=build_synthesis_prompt(inputs["query"], results)),
    ]


class LangChainRAG:
    """Chaîne LCEL : retriever → prompt → modèle de chat en sortie structurée.

    Args:
        store: vector store interrogé (via `VectorStoreRetriever`).
        chat_model: modèle de chat LangChain supportant `with_structured_output`.
        model_name: identifiant du modèle, pour le journal de coût.
        ledger: journal de coût alimenté à chaque appel (optionnel).
    """

    name = "langchain"

    def __init__(
        self,
        store: VectorStore,
        chat_model: BaseChatModel,
        model_name: str,
        ledger: CostLedger | None = None,
    ) -> None:
        self._store = store
        self._model_name = model_name
        self._ledger = ledger
        self._structured: Runnable[Any, Any] = chat_model.with_structured_output(
            GroundedAnswer, include_raw=True
        )

    def chain(self, k: int) -> Runnable[dict[str, Any], dict[str, Any]]:
        """La chaîne LCEL pour `k` chunks : entrée `{"query"}`, sortie avec `llm_output`."""
        retriever = VectorStoreRetriever(store=self._store, k=k)
        return (
            RunnablePassthrough.assign(
                results=itemgetter("query")
                | retriever
                | RunnableLambda(documents_to_results)
            )
            | RunnablePassthrough.assign(
                llm_output=RunnableLambda(_messages) | self._structured
            )
        ).with_config(run_name="langchain_rag")

    def answer(self, query: str, k: int = 5) -> AnswerResult:
        """Réponse citée à `query`.

        Raises:
            EmptyQueryError: requête vide.
            NoRelevantChunkError: aucun chunk retrouvé.
            LLMOutputError: sortie du modèle invalide après la nouvelle tentative.
            LLMGenerationError: erreur d'API.
        """
        if not query.strip():
            raise EmptyQueryError("La requête utilisateur est vide.")

        chain = self.chain(k)
        with trace_span("langchain.pipeline", k=k):
            for attempt in range(_OUTPUT_RETRIES + 1):
                try:
                    out = chain.invoke({"query": query})
                except anthropic.APIError as exc:
                    raise LLMGenerationError(
                        f"L'appel à l'API Anthropic a échoué : {exc}"
                    ) from exc
                self._record_usage(out["llm_output"]["raw"])
                try:
                    grounded = self._parse(out["llm_output"])
                except LLMOutputError:
                    if attempt == _OUTPUT_RETRIES:
                        raise
                    logger.warning("langchain.structured_retry", attempt=attempt + 1)
                    continue
                return build_answer(query, out["results"], grounded)
        raise AssertionError("inatteignable")  # pragma: no cover

    @staticmethod
    def _parse(llm_output: dict[str, Any]) -> GroundedAnswer:
        parsed = llm_output.get("parsed")
        if isinstance(parsed, GroundedAnswer):
            return parsed
        # Même réparation que les clients du projet (liste sérialisée en chaîne...).
        tool_calls = getattr(llm_output.get("raw"), "tool_calls", None) or []
        if not tool_calls:
            raise LLMOutputError("Le modèle n'a renvoyé aucun appel d'outil.")
        return validate_tool_input(GroundedAnswer, tool_calls[0]["args"])

    def _record_usage(self, raw: Any) -> None:
        usage = getattr(raw, "usage_metadata", None) or {}
        if self._ledger is not None:
            self._ledger.record(
                LLMCallRecord(
                    role=LLMRole.SYNTHESIS.value,
                    model=self._model_name,
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)),
                    cached=False,
                )
            )


def build_chat_model(api_key: str, model: str) -> "ChatAnthropic":
    """`ChatAnthropic` réglé comme `AnthropicLLMClient` (modèle, max_tokens)."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(  # type: ignore[call-arg]
        model_name=model, api_key=SecretStr(api_key), max_tokens=DEFAULT_MAX_TOKENS
    )


def build_langchain_rag(
    store: VectorStore, ledger: CostLedger | None = None
) -> LangChainRAG:
    """Backend LangChain configuré depuis les settings (fournisseur Anthropic uniquement).

    Raises:
        ValueError: fournisseur autre qu'Anthropic, ou clé absente.
    """
    if settings.llm_provider.strip().lower() != "anthropic":
        raise ValueError(
            "RAG_BACKEND=langchain ne supporte que LLM_PROVIDER=anthropic."
        )
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY absente : backend LangChain indisponible.")
    model = model_for(LLMRole.SYNTHESIS)
    return LangChainRAG(
        store, build_chat_model(settings.anthropic_api_key, model), model, ledger
    )
