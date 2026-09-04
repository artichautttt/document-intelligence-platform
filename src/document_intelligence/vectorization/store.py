"""Interface abstraite commune à toute implémentation de vector store.

Cette abstraction isole le reste du pipeline (chunking, orchestration, API)
du vector store concret utilisé. ChromaDB est l'implémentation de
développement (Sprint 2) ; une migration future vers Qdrant ou Pinecone se
fera en ajoutant une nouvelle classe `VectorStore` sans modifier les
appelants.
"""

from abc import ABC, abstractmethod

from document_intelligence.chunking.models import Chunk
from document_intelligence.vectorization.models import QueryResult


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
