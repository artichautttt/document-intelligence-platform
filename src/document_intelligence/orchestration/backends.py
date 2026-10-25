"""Backends de réponse RAG interchangeables, sélectionnés par `RAG_BACKEND`.

- `langgraph` (défaut) : le graphe existant (`orchestration.graph.answer_query`).
- `langchain` : chaîne LCEL (`orchestration.langchain_backend`), groupe de
  dépendances optionnel `langchain`.

Les deux partagent le vector store, les prompts de synthèse et la conversion
de la sortie du modèle en réponse citée (`nodes.build_answer`) : seul le
framework d'orchestration change.
"""

from dataclasses import dataclass
from typing import Protocol

from document_intelligence.orchestration.graph import answer_query
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.orchestration.models import AnswerResult
from document_intelligence.orchestration.verification import AnswerVerifier
from document_intelligence.vectorization.store import VectorStore

BACKENDS = ("langgraph", "langchain")


class RAGBackend(Protocol):
    """Répond à une question à partir des `k` chunks les plus pertinents."""

    name: str

    def answer(self, query: str, k: int = 5) -> AnswerResult: ...


@dataclass
class LangGraphBackend:
    """Le pipeline historique : graphe LangGraph routage → retrieval → synthèse."""

    store: VectorStore
    llm: LLMClient
    verifier: AnswerVerifier | None = None
    max_regenerations: int = 1
    name: str = "langgraph"

    def answer(self, query: str, k: int = 5) -> AnswerResult:
        return answer_query(
            query,
            store=self.store,
            llm=self.llm,
            k=k,
            verifier=self.verifier,
            max_regenerations=self.max_regenerations,
        )


def check_backend(name: str) -> str:
    """Nom de backend normalisé.

    Raises:
        ValueError: si `name` ne désigne pas un backend supporté.
    """
    normalized = name.strip().lower()
    if normalized not in BACKENDS:
        raise ValueError(
            f"RAG_BACKEND={name!r} non supporté ; valeurs possibles : {', '.join(BACKENDS)}."
        )
    return normalized
