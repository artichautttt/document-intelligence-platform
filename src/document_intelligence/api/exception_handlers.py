"""Traduction des exceptions métier typées en réponses HTTP explicites.

Chaque module (`ingestion`, `vectorization`, `orchestration`) lève des
exceptions dédiées plutôt que de laisser échouer silencieusement ; ce module
les mappe vers des codes HTTP pour que l'appelant de l'API dispose de la
même granularité d'erreur qu'un appelant de la librairie.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from document_intelligence.ingestion.exceptions import (
    CorruptFileError,
    EmptyDocumentError,
    UnsupportedFormatError,
)
from document_intelligence.orchestration.exceptions import (
    EmptyQueryError,
    LLMGenerationError,
    NoRelevantChunkError,
)
from document_intelligence.vectorization.exceptions import (
    EmptyChunkListError,
    VectorStoreConnectionError,
)


def _error_response(status_code: int, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


def register_exception_handlers(app: FastAPI) -> None:
    """Enregistre les handlers traduisant les exceptions métier en réponses HTTP."""

    @app.exception_handler(UnsupportedFormatError)
    async def _unsupported_format(
        request: Request, exc: UnsupportedFormatError
    ) -> JSONResponse:
        return _error_response(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, exc)

    @app.exception_handler(CorruptFileError)
    async def _corrupt_file(request: Request, exc: CorruptFileError) -> JSONResponse:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, exc)

    @app.exception_handler(EmptyDocumentError)
    async def _empty_document(
        request: Request, exc: EmptyDocumentError
    ) -> JSONResponse:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, exc)

    @app.exception_handler(EmptyChunkListError)
    async def _empty_chunk_list(
        request: Request, exc: EmptyChunkListError
    ) -> JSONResponse:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, exc)

    @app.exception_handler(VectorStoreConnectionError)
    async def _vector_store_connection(
        request: Request, exc: VectorStoreConnectionError
    ) -> JSONResponse:
        return _error_response(status.HTTP_503_SERVICE_UNAVAILABLE, exc)

    @app.exception_handler(EmptyQueryError)
    async def _empty_query(request: Request, exc: EmptyQueryError) -> JSONResponse:
        return _error_response(status.HTTP_400_BAD_REQUEST, exc)

    @app.exception_handler(NoRelevantChunkError)
    async def _no_relevant_chunk(
        request: Request, exc: NoRelevantChunkError
    ) -> JSONResponse:
        return _error_response(status.HTTP_404_NOT_FOUND, exc)

    @app.exception_handler(LLMGenerationError)
    async def _llm_generation(
        request: Request, exc: LLMGenerationError
    ) -> JSONResponse:
        return _error_response(status.HTTP_502_BAD_GATEWAY, exc)
