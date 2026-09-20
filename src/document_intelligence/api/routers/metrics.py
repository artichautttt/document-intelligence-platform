"""Endpoint d'observabilité : latence par étape, tokens et coût estimé des appels LLM."""

from typing import Any

from fastapi import APIRouter, Depends

from document_intelligence.api.dependencies import verify_api_key
from document_intelligence.observability import registry

router = APIRouter(tags=["observability"], dependencies=[Depends(verify_api_key)])


@router.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    """Retourne un instantané des métriques collectées depuis le démarrage."""
    return registry.snapshot()
