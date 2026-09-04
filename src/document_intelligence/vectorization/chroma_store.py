"""Implémentation `VectorStore` basée sur ChromaDB.

Utilise la fonction d'embedding par défaut de ChromaDB (all-MiniLM-L6-v2 via
onnxruntime) : un modèle léger (~90 Mo, pas de dépendance torch), suffisant
pour un environnement de développement et adapté aux contraintes d'espace
disque du poste actuel. La collection est configurée en espace cosinus, ce
qui permet de dériver un score de similarité simple (`1 - distance`).
"""

import json

import chromadb

from document_intelligence.chunking.models import Chunk, ChunkProvenance
from document_intelligence.core.logging import get_logger
from document_intelligence.vectorization.exceptions import EmptyChunkListError
from document_intelligence.vectorization.models import QueryResult
from document_intelligence.vectorization.store import VectorStore

logger = get_logger(__name__)

_DEFAULT_COLLECTION_NAME = "document_chunks"


class ChromaStore(VectorStore):
    """Vector store de développement, persisté localement via ChromaDB."""

    def __init__(
        self,
        persist_directory: str = ".chroma",
        collection_name: str = _DEFAULT_COLLECTION_NAME,
    ) -> None:
        client = chromadb.PersistentClient(path=persist_directory)
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise EmptyChunkListError("Impossible de vectoriser une liste de chunks vide")

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._to_metadata(chunk.provenance) for chunk in chunks],
        )
        logger.info("vectorization.add_chunks", chunk_count=len(chunks))

    def query(self, text: str, k: int = 5) -> list[QueryResult]:
        result = self._collection.query(query_texts=[text], n_results=k)

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            QueryResult(
                chunk_id=chunk_id,
                text=document,
                score=1.0 - distance,
                provenance=self._from_metadata(metadata),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    @staticmethod
    def _to_metadata(provenance: ChunkProvenance) -> dict[str, str]:
        return {
            "document_id": provenance.document_id,
            "source_path": provenance.source_path,
            "element_ids": json.dumps(provenance.element_ids),
        }

    @staticmethod
    def _from_metadata(metadata: dict) -> ChunkProvenance:
        return ChunkProvenance(
            document_id=metadata["document_id"],
            source_path=metadata["source_path"],
            element_ids=json.loads(metadata["element_ids"]),
        )
