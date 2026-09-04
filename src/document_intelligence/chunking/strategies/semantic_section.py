"""Stratégie de chunking structurel basé sur les sections du document.

Logique retenue (chunking sémantique structurel, sans découpage naïf par
nombre de caractères fixe) :

1. Le document est d'abord segmenté en *sections* : chaque élément `TITLE`
   ouvre une nouvelle section, qui regroupe tous les éléments suivants
   jusqu'au prochain `TITLE`. Le chunking respecte ainsi les frontières de
   sections au lieu de trancher le texte à un offset arbitraire.
2. À l'intérieur d'une section, les éléments `TABLE` sont toujours émis en
   chunks dédiés : fusionner un tableau avec du texte narratif briserait sa
   structure et sa lisibilité pour un LLM en aval.
3. Les éléments de texte narratif/liste sont accumulés dans un chunk tant
   que la limite `max_chunk_chars` n'est pas dépassée. Seule une section trop
   longue pour tenir dans un chunk est sous-découpée, et uniquement sur des
   frontières de phrases (jamais au milieu d'une phrase), pour préserver le
   sens local du texte.

Chaque chunk produit référence les `element_id` exacts qui l'ont composé,
garantissant la traçabilité réponse -> chunk -> élément source.
"""

import re
import uuid

from document_intelligence.chunking.models import Chunk, ChunkProvenance
from document_intelligence.chunking.strategies.base import ChunkingStrategy
from document_intelligence.ingestion.models import DocumentElement, ElementType, ParsedDocument

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class SemanticSectionChunker(ChunkingStrategy):
    """Découpe un document en respectant ses frontières de sections et de phrases."""

    def __init__(self, max_chunk_chars: int = 1500) -> None:
        self._max_chunk_chars = max_chunk_chars

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in self._split_into_sections(document.elements):
            chunks.extend(self._chunk_section(section, document))
        return chunks

    def _split_into_sections(
        self, elements: list[DocumentElement]
    ) -> list[list[DocumentElement]]:
        sections: list[list[DocumentElement]] = []
        current: list[DocumentElement] = []

        for element in elements:
            if element.element_type is ElementType.TITLE and current:
                sections.append(current)
                current = []
            current.append(element)

        if current:
            sections.append(current)

        return sections

    def _chunk_section(
        self, section: list[DocumentElement], document: ParsedDocument
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer_elements: list[DocumentElement] = []
        buffer_text = ""

        def flush() -> None:
            nonlocal buffer_elements, buffer_text
            if buffer_elements:
                chunks.append(self._make_chunk(buffer_elements, buffer_text, document))
            buffer_elements = []
            buffer_text = ""

        for element in section:
            if element.element_type is ElementType.TABLE:
                flush()
                chunks.append(self._make_chunk([element], element.text, document))
                continue

            for piece in self._split_if_needed(element):
                candidate = f"{buffer_text}\n{piece}".strip() if buffer_text else piece
                if len(candidate) > self._max_chunk_chars and buffer_text:
                    flush()
                    candidate = piece

                buffer_text = candidate
                if not buffer_elements or buffer_elements[-1] is not element:
                    buffer_elements.append(element)

        flush()
        return chunks

    def _split_if_needed(self, element: DocumentElement) -> list[str]:
        """Découpe le texte d'un élément trop long sur des frontières de phrases."""
        if len(element.text) <= self._max_chunk_chars:
            return [element.text]

        sentences = _SENTENCE_BOUNDARY.split(element.text)
        fragments: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > self._max_chunk_chars and current:
                fragments.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            fragments.append(current)

        return fragments

    def _make_chunk(
        self, elements: list[DocumentElement], text: str, document: ParsedDocument
    ) -> Chunk:
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            text=text.strip(),
            provenance=ChunkProvenance(
                document_id=document.document_id,
                source_path=document.source_path,
                element_ids=[el.element_id for el in elements],
            ),
        )
