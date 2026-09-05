"""Point d'entrée public du module de chunking."""

from document_intelligence.chunking.models import Chunk
from document_intelligence.chunking.strategies.base import ChunkingStrategy
from document_intelligence.chunking.strategies.semantic_section import (
    SemanticSectionChunker,
)
from document_intelligence.core.config import settings
from document_intelligence.core.logging import get_logger
from document_intelligence.ingestion.models import ParsedDocument

logger = get_logger(__name__)


def chunk_document(
    document: ParsedDocument, strategy: ChunkingStrategy | None = None
) -> list[Chunk]:
    """Découpe un document parsé en chunks traçables via la stratégie fournie.

    Args:
        document: document préalablement produit par le module d'ingestion.
        strategy: stratégie de chunking à utiliser (par défaut `SemanticSectionChunker`).
    """
    active_strategy = (
        strategy
        if strategy is not None
        else SemanticSectionChunker(max_chunk_chars=settings.max_chunk_chars)
    )

    chunks = active_strategy.chunk(document)
    logger.info(
        "chunking.success",
        document_id=document.document_id,
        chunk_count=len(chunks),
        strategy=type(active_strategy).__name__,
    )
    return chunks
