"""Implémentation `VectorStore` basée sur Qdrant.

Contrairement à ChromaDB, Qdrant ne fournit pas de fonction d'embedding
intégrée : on réutilise donc la fonction d'embedding par défaut de ChromaDB
(all-MiniLM-L6-v2 via onnxruntime, déjà une dépendance du projet) pour
vectoriser les textes avant de les stocker, ce qui évite d'introduire une
dépendance supplémentaire (torch, sentence-transformers...) uniquement pour
ce backend. Les identifiants de chunk sont des UUID4 (cf. `chunking`), donc
directement utilisables comme identifiants de point Qdrant.
"""

import json

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from document_intelligence.chunking.models import Chunk, ChunkProvenance
from document_intelligence.core.logging import get_logger
from document_intelligence.vectorization.exceptions import EmptyChunkListError
from document_intelligence.vectorization.models import QueryResult
from document_intelligence.vectorization.store import VectorStore

logger = get_logger(__name__)

_DEFAULT_COLLECTION_NAME = "document_chunks"


class QdrantStore(VectorStore):
    """Vector store de production, s'appuyant sur un serveur Qdrant dédié."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = _DEFAULT_COLLECTION_NAME,
        location: str | None = None,
    ) -> None:
        """Initialise le client Qdrant.

        Args:
            url: URL du serveur Qdrant à utiliser (ignoré si `location` est fourni).
            collection_name: nom de la collection Qdrant.
            location: réservé aux tests (`":memory:"` pour un Qdrant embarqué en
                mémoire, sans serveur externe).
        """
        self._client = (
            QdrantClient(location=location) if location else QdrantClient(url=url)
        )
        self._collection_name = collection_name
        self._embed = DefaultEmbeddingFunction()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection_name):
            return

        vector_size = len(self._embed(["_probe_"])[0])
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise EmptyChunkListError(
                "Impossible de vectoriser une liste de chunks vide"
            )

        vectors = self._embed([chunk.text for chunk in chunks])
        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=[float(x) for x in vector],
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    **self._to_metadata(chunk.provenance),
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)
        logger.info(
            "vectorization.add_chunks", chunk_count=len(chunks), backend="qdrant"
        )

    def query(self, text: str, k: int = 5) -> list[QueryResult]:
        vector = self._embed([text])[0]
        results = self._client.query_points(
            collection_name=self._collection_name, query=vector, limit=k
        ).points

        query_results = []
        for point in results:
            payload = point.payload or {}
            query_results.append(
                QueryResult(
                    chunk_id=payload["chunk_id"],
                    text=payload["text"],
                    score=point.score,
                    provenance=self._from_metadata(payload),
                )
            )
        return query_results

    @staticmethod
    def _to_metadata(provenance: ChunkProvenance) -> dict[str, str]:
        return {
            "document_id": provenance.document_id,
            "source_path": provenance.source_path,
            "element_ids": json.dumps(provenance.element_ids),
        }

    @staticmethod
    def _from_metadata(payload: dict[str, str]) -> ChunkProvenance:
        return ChunkProvenance(
            document_id=payload["document_id"],
            source_path=payload["source_path"],
            element_ids=json.loads(payload["element_ids"]),
        )
