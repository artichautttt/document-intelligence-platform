"""Doublures partagées par les tests des outils et du serveur MCP."""

import re
import uuid

import pytest

from document_intelligence.chunking.models import Chunk, ChunkProvenance
from document_intelligence.orchestration.llm import LLMClient, T
from document_intelligence.vectorization.models import DocumentSummary, QueryResult
from document_intelligence.vectorization.store import VectorStore, summarize_documents


def make_chunk(
    text: str, document_id: str = "doc-1", source: str = "/docs/a.pdf"
) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        text=text,
        provenance=ChunkProvenance(
            document_id=document_id, source_path=source, element_ids=["el-1"]
        ),
    )


class InMemoryStore(VectorStore):
    """Store en mémoire : renvoie les chunks dans l'ordre d'insertion."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.queries: list[tuple[str, int]] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def query(self, text: str, k: int = 5) -> list[QueryResult]:
        self.queries.append((text, k))
        return [
            QueryResult(
                chunk_id=c.chunk_id, text=c.text, score=0.9, provenance=c.provenance
            )
            for c in self.chunks[:k]
        ]

    def list_documents(self) -> list[DocumentSummary]:
        return summarize_documents(
            [(c.provenance.document_id, c.provenance.source_path) for c in self.chunks]
        )


class CitingLLM(LLMClient):
    """LLM déterministe : cite le premier extrait de son prompt."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> str:
        return "non utilisé"

    def complete_structured(
        self, prompt: str, schema: type[T], system: str | None = None
    ) -> T:
        self.calls += 1
        ids = re.findall(r'<extrait id="([^"]+)"', prompt)
        return schema.model_validate(
            {
                "answer": "La redevance est de 184 000 euros HT.",
                "cited_chunk_ids": ids[:1],
                "sufficient_context": True,
            }
        )


def demo_store() -> InMemoryStore:
    """Deux documents, trois chunks ; le premier contient la redevance."""
    return InMemoryStore(
        [
            make_chunk(
                "La redevance annuelle est de 184 000 euros HT.",
                "doc-1",
                "/docs/contrat.pdf",
            ),
            make_chunk(
                "Le préavis est de quatre-vingt-dix jours.",
                "doc-1",
                "/docs/contrat.pdf",
            ),
            make_chunk(
                "Le chiffre d'affaires progresse de 12 %.", "doc-2", "/docs/rapport.pdf"
            ),
        ]
    )


@pytest.fixture
def store() -> InMemoryStore:
    return demo_store()
