"""Mise en quarantaine, à l'ingestion, des chunks porteurs d'une injection."""

from dataclasses import dataclass, field

from document_intelligence.chunking.models import Chunk
from document_intelligence.core.logging import get_logger
from document_intelligence.guardrails.rules import detect

logger = get_logger(__name__)


@dataclass(frozen=True)
class QuarantinedChunk:
    chunk: Chunk
    rules: list[str]


@dataclass
class ScreeningResult:
    kept: list[Chunk] = field(default_factory=list)
    quarantined: list[QuarantinedChunk] = field(default_factory=list)


def screen_chunks(chunks: list[Chunk]) -> ScreeningResult:
    """Sépare les chunks sains de ceux qui déclenchent une règle (cible `document`).

    Les chunks écartés ne sont pas indexés ; chacun est journalisé
    (`guardrails.chunk_quarantined`) avec sa source et les règles déclenchées.
    """
    result = ScreeningResult()
    for chunk in chunks:
        matches = detect(chunk.text, "document")
        if not matches:
            result.kept.append(chunk)
            continue
        rules = [m.rule for m in matches]
        result.quarantined.append(QuarantinedChunk(chunk, rules))
        logger.warning(
            "guardrails.chunk_quarantined",
            chunk_id=chunk.chunk_id,
            source_path=chunk.provenance.source_path,
            rules=rules,
        )
    return result
