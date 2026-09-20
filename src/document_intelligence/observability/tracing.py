"""Tracing léger des étapes du pipeline : latence, erreurs et attributs par span.

Chaque `trace_span` émet un log structuré et notifie les `SpanSink`
enregistrés. Le registre de métriques interne (`observability.metrics`) et
l'export OpenTelemetry (`observability.otel`) en sont deux ; un autre backend
se branche en implémentant `SpanSink`, sans toucher au code instrumenté.

L'observabilité ne doit jamais casser le pipeline : une exception levée par
un sink est journalisée puis ignorée.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from document_intelligence.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SpanRecord:
    """Trace d'une étape terminée."""

    name: str
    duration_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class SpanSink(ABC):
    """Destination des spans (métriques, backend de tracing externe...)."""

    def on_span_start(  # noqa: B027 - hook facultatif : implémentation vide voulue
        self, name: str, attributes: dict[str, Any]
    ) -> None:
        """Appelé à l'ouverture d'un span (facultatif ; utile pour la hiérarchie)."""

    @abstractmethod
    def on_span(self, record: SpanRecord) -> None:
        """Reçoit un span terminé."""


_sinks: list[SpanSink] = []


def add_sink(sink: SpanSink) -> None:
    """Enregistre un sink (idempotent)."""
    if sink not in _sinks:
        _sinks.append(sink)


def remove_sink(sink: SpanSink) -> None:
    """Retire un sink s'il est enregistré."""
    if sink in _sinks:
        _sinks.remove(sink)


def _notify(action: Callable[[SpanSink], None]) -> None:
    for sink in list(_sinks):
        try:
            action(sink)
        except Exception:
            logger.warning("span.sink_failed", sink=type(sink).__name__, exc_info=True)


@contextmanager
def trace_span(name: str, **attributes: Any) -> Iterator[dict[str, Any]]:
    """Mesure la durée du bloc et enregistre le span, y compris en cas d'erreur.

    Le dictionnaire yieldé permet d'ajouter des attributs calculés dans le
    bloc (nombre de résultats, score...). L'exception éventuelle est relancée.
    """
    span_attributes: dict[str, Any] = dict(attributes)
    error: str | None = None
    _notify(lambda sink: sink.on_span_start(name, dict(attributes)))
    start = time.perf_counter()
    try:
        yield span_attributes
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        record = SpanRecord(
            name=name,
            duration_ms=(time.perf_counter() - start) * 1000,
            attributes=span_attributes,
            error=error,
        )
        logger.info(
            "span",
            span=name,
            duration_ms=round(record.duration_ms, 2),
            error=error,
            **span_attributes,
        )
        _notify(lambda sink: sink.on_span(record))
