"""Interface abstraite commune à toute implémentation de vector store.

Cette abstraction isole le reste du pipeline (chunking, orchestration, API)
du vector store concret utilisé. ChromaDB est l'implémentation de
développement (Sprint 2) ; une migration future vers Qdrant ou Pinecone se
fera en ajoutant une nouvelle classe `VectorStore` sans modifier les
appelants.
"""

from abc import ABC, abstractmethod

from document_intelligence.chunking.models import Chunk
from document_intelligence.vectorization.models import DocumentSummary, QueryResult


class VectorStore(ABC):
    """Contrat que doit respecter tout backend de stockage et recherche vectorielle."""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Vectorise et persiste une liste de chunks.

        Raises:
            EmptyChunkListError: si `chunks` est vide.
        """

    @abstractmethod
    def query(self, text: str, k: int = 5) -> list[QueryResult]:
        """Retourne les `k` chunks les plus similaires sémantiquement à `text`."""

    def list_documents(self) -> list[DocumentSummary]:
        """Documents indexés, triés par `source_path`, avec leur nombre de chunks.

        Non abstraite pour ne pas imposer cette méthode aux implémentations
        existantes (doublures de test comprises) : un backend qui ne la fournit
        pas lève `NotImplementedError`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} ne sait pas lister les documents indexés."
        )


def summarize_documents(provenances: list[tuple[str, str]]) -> list[DocumentSummary]:
    """Regroupe des couples (document_id, source_path), un par chunk, par document."""
    counts: dict[tuple[str, str], int] = {}
    for key in provenances:
        counts[key] = counts.get(key, 0) + 1
    return sorted(
        (
            DocumentSummary(document_id=doc_id, source_path=path, chunk_count=count)
            for (doc_id, path), count in counts.items()
        ),
        key=lambda summary: (summary.source_path, summary.document_id),
    )
