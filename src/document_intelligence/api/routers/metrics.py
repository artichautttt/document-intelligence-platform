"""Endpoint d'observabilité : latence par étape, tokens et coût estimé des appels LLM."""

from typing import Any

from fastapi import APIRouter, Depends, Response

from document_intelligence.api.dependencies import verify_api_key
from document_intelligence.observability import registry
from document_intelligence.observability.prometheus import (
    CONTENT_TYPE_LATEST,
    render_latest,
)

router = APIRouter(tags=["observability"], dependencies=[Depends(verify_api_key)])


@router.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    """Retourne un instantané des métriques collectées depuis le démarrage."""
    return registry.snapshot()


@router.get("/metrics/prometheus")
async def get_prometheus_metrics() -> Response:
    """Métriques au format texte Prometheus (scrapées par le service `prometheus`)."""
    return Response(content=render_latest(), media_type=CONTENT_TYPE_LATEST)
