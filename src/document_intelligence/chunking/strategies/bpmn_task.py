"""Chunking des modèles de processus BPMN : un chunk par tâche.

Chaque tâche décrite par `BpmnParser` (acteur, entrées, sorties, décision)
devient un chunk, précédé de la ligne d'en-tête de son processus
(« Processus : ... »), pour que le chunk reste compréhensible seul.

Citation processus > tâche : `element_ids` vaut `[id du processus, id de la
tâche]`, les identifiants BPMN du fichier. Une réponse qui cite le chunk
renvoie donc au processus puis à la tâche exacte.

Une tâche n'est jamais découpée : sa description tient en quelques lignes,
et la couper séparerait l'acteur de la décision.
"""

import uuid

from document_intelligence.chunking.models import Chunk, ChunkProvenance
from document_intelligence.chunking.strategies.base import ChunkingStrategy
from document_intelligence.ingestion.models import (
    DocumentElement,
    ElementType,
    ParsedDocument,
)


class BpmnTaskChunker(ChunkingStrategy):
    """Un chunk par élément `TASK`, rattaché au dernier élément `PROCESS` vu."""

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        process: DocumentElement | None = None
        for element in document.elements:
            if element.element_type is ElementType.PROCESS:
                process = element
            elif element.element_type is ElementType.TASK:
                header = process.text.splitlines()[0] if process else ""
                element_ids = (
                    [process.element_id, element.element_id]
                    if process
                    else [element.element_id]
                )
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=f"{header}\n{element.text}".strip(),
                        provenance=ChunkProvenance(
                            document_id=document.document_id,
                            source_path=document.source_path,
                            element_ids=element_ids,
                        ),
                    )
                )
        return chunks
