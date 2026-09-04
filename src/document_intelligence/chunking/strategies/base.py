"""Interface abstraite commune à toutes les stratégies de chunking."""

from abc import ABC, abstractmethod

from document_intelligence.chunking.models import Chunk
from document_intelligence.ingestion.models import ParsedDocument


class ChunkingStrategy(ABC):
    """Contrat que doit respecter toute stratégie transformant un document parsé en chunks."""

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Découpe un document parsé en chunks, en conservant leur provenance exacte."""
