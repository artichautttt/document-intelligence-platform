"""Observabilité : tracing des étapes du pipeline et métriques LLM (latence, tokens, coût)."""

from document_intelligence.observability.metrics import MetricsRegistry, registry
from document_intelligence.observability.store import SqliteMetricsStore
from document_intelligence.observability.tracing import (
    SpanRecord,
    SpanSink,
    add_sink,
    remove_sink,
    trace_span,
)

# Le registre interne collecte tous les spans dès l'import du package.
add_sink(registry)

__all__ = [
    "MetricsRegistry",
    "SpanRecord",
    "SpanSink",
    "SqliteMetricsStore",
    "add_sink",
    "registry",
    "remove_sink",
    "trace_span",
]
