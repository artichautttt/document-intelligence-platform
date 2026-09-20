"""Export des spans vers un backend OpenTelemetry (OTLP/HTTP).

Compatible avec tout collecteur OTLP (Jaeger, Tempo, Grafana...) et avec
Langfuse, qui expose un endpoint OTLP. Nécessite le groupe de dépendances
`otel` (`uv sync --group otel`) ; sans lui, `setup_otel` lève `ImportError`.

La hiérarchie des spans (pipeline → nœuds → appel LLM) est reconstruite grâce
au hook `on_span_start` : chaque span OpenTelemetry est ouvert dans le contexte
courant, puis fermé avec ses attributs à la fin du `trace_span` correspondant.
"""

from contextvars import ContextVar
from typing import Any

from opentelemetry import context, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

from document_intelligence.observability.tracing import (
    SpanRecord,
    SpanSink,
    add_sink,
)

_TRACES_PATH = "/v1/traces"


def _coerce(value: Any) -> str | bool | int | float | None:
    if value is None or isinstance(value, str | bool | int | float):
        return value
    return str(value)


def _set_attributes(span: Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        coerced = _coerce(value)
        if coerced is not None:
            span.set_attribute(key, coerced)


def parse_headers(raw: str | None) -> dict[str, str]:
    """Parse `k1=v1,k2=v2` (format de `OTEL_EXPORTER_OTLP_HEADERS`)."""
    if not raw:
        return {}
    pairs = (item.split("=", 1) for item in raw.split(",") if "=" in item)
    return {key.strip(): value.strip() for key, value in pairs}


def traces_endpoint(base: str) -> str:
    """Ajoute `/v1/traces` à un endpoint OTLP de base s'il n'y figure pas déjà.

    L'exporteur n'ajoute ce suffixe que lorsqu'il lit lui-même la variable
    d'environnement ; un `endpoint=` explicite est utilisé tel quel.
    """
    base = base.rstrip("/")
    return base if base.endswith(_TRACES_PATH) else base + _TRACES_PATH


class OtelSink(SpanSink):
    """Traduit chaque `trace_span` en span OpenTelemetry."""

    def __init__(self, provider: TracerProvider) -> None:
        self._provider = provider
        self._tracer = provider.get_tracer("document_intelligence")
        self._stack: ContextVar[tuple[tuple[Span, object], ...]] = ContextVar(
            "otel_span_stack", default=()
        )

    def on_span_start(self, name: str, attributes: dict[str, Any]) -> None:
        span = self._tracer.start_span(name)
        _set_attributes(span, attributes)
        token = context.attach(trace.set_span_in_context(span))
        self._stack.set((*self._stack.get(), (span, token)))

    def on_span(self, record: SpanRecord) -> None:
        stack = self._stack.get()
        if not stack:
            return
        (span, token), remaining = stack[-1], stack[:-1]
        self._stack.set(remaining)
        _set_attributes(span, record.attributes)
        if record.error is not None:
            span.set_status(Status(StatusCode.ERROR, record.error))
        span.end()
        context.detach(token)  # type: ignore[arg-type]

    def shutdown(self) -> None:
        """Vide les spans en attente d'export et arrête le fournisseur."""
        self._provider.shutdown()


def setup_otel(
    endpoint: str | None = None,
    headers: str | None = None,
    service_name: str = "document-intelligence",
    processor: SpanProcessor | None = None,
) -> OtelSink:
    """Active l'export OpenTelemetry et enregistre le sink de tracing.

    Args:
        endpoint: endpoint OTLP/HTTP de base (ex. `https://host/api/public/otel`).
        headers: en-têtes d'authentification, format `k1=v1,k2=v2`.
        service_name: nom de service déclaré dans la ressource OTel.
        processor: processeur de spans à utiliser à la place de l'export OTLP
            (utile pour les tests, avec un exporteur en mémoire).
    """
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if processor is None:
        exporter = OTLPSpanExporter(
            endpoint=traces_endpoint(endpoint) if endpoint else None,
            headers=parse_headers(headers) or None,
        )
        processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    sink = OtelSink(provider)
    add_sink(sink)
    return sink
