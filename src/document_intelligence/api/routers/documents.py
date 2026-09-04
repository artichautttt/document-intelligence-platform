"""Endpoints d'ingestion : upload d'un document, chunking, vectorisation."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from document_intelligence.api.dependencies import get_store
from document_intelligence.api.schemas import DocumentIngestResponse
from document_intelligence.chunking.service import chunk_document
from document_intelligence.core.logging import get_logger
from document_intelligence.ingestion.service import ingest
from document_intelligence.vectorization.store import VectorStore

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentIngestResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    store: VectorStore = Depends(get_store),
) -> DocumentIngestResponse:
    """Ingère, découpe et vectorise le document envoyé, puis retourne son identifiant.

    Le fichier est écrit dans un répertoire temporaire le temps du traitement :
    les parsers de `ingestion/` opèrent sur des chemins de fichiers, pas sur
    des flux en mémoire.
    """
    suffix = Path(file.filename or "").suffix
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"upload{suffix}"
        tmp_path.write_bytes(await file.read())

        document = ingest(tmp_path)
        chunks = chunk_document(document)
        store.add_chunks(chunks)

    logger.info("api.document_ingested", document_id=document.document_id, chunk_count=len(chunks))
    return DocumentIngestResponse(document_id=document.document_id, chunk_count=len(chunks))
