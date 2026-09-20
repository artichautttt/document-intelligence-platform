"""Tests de la persistance SQLite des métriques et de l'export OpenTelemetry."""

from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from document_intelligence.observability import (
    MetricsRegistry,
    SpanRecord,
    SpanSink,
    SqliteMetricsStore,
    add_sink,
    remove_sink,
    trace_span,
)
from document_intelligence.observability.otel import (
    OtelSink,
    parse_headers,
    setup_otel,
    traces_endpoint,
)


class TestSqliteMetricsStore:
    def test_registry_state_survives_restart(self, tmp_path: Path) -> None:
        db = tmp_path / "metrics.db"

        first = MetricsRegistry()
        store = SqliteMetricsStore(db)
        first.attach_store(store)
        for duration in (10.0, 20.0, 30.0):
            first.on_span(SpanRecord(name="node.retrieve", duration_ms=duration))
        first.on_span(SpanRecord(name="node.retrieve", duration_ms=40.0, error="X"))
        first.record_llm_call(input_tokens=100, output_tokens=40, latency_ms=250)
        expected = first.snapshot()
        store.close()

        second = MetricsRegistry()
        reopened = SqliteMetricsStore(db)
        second.attach_store(reopened)
        restored = second.snapshot()
        reopened.close()

        assert restored == expected
        assert restored["spans"]["node.retrieve"]["count"] == 4
        assert restored["spans"]["node.retrieve"]["errors"] == 1
        assert restored["llm"]["input_tokens"] == 100
        assert restored["persistent"] is True

    def test_new_events_accumulate_on_top_of_restored_history(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "metrics.db"
        store = SqliteMetricsStore(db)
        first = MetricsRegistry()
        first.attach_store(store)
        first.on_span(SpanRecord(name="pipeline", duration_ms=5.0))
        store.close()

        store2 = SqliteMetricsStore(db)
        second = MetricsRegistry()
        second.attach_store(store2)
        second.on_span(SpanRecord(name="pipeline", duration_ms=7.0))

        assert second.snapshot()["spans"]["pipeline"]["count"] == 2
        store2.close()

    def test_registry_without_store_is_not_persistent(self) -> None:
        assert MetricsRegistry().snapshot()["persistent"] is False

    def test_events_older_than_retention_are_purged(self, tmp_path: Path) -> None:
        db = tmp_path / "metrics.db"
        store = SqliteMetricsStore(db)
        store.append_span("old", 1.0, None)
        store._conn.execute("UPDATE span_events SET ts = ts - 100 * 86400")
        store._conn.commit()
        store.close()

        reopened = SqliteMetricsStore(db, retention_days=30)

        assert reopened.span_totals() == {}
        reopened.close()

    def test_write_failure_never_raises(self, tmp_path: Path) -> None:
        store = SqliteMetricsStore(tmp_path / "metrics.db")
        store._conn.close()

        store.append_span("x", 1.0, None)  # ne doit pas lever


class TestSinkResilience:
    def test_failing_sink_does_not_break_the_traced_code(self) -> None:
        class Broken(SpanSink):
            def on_span_start(self, name: str, attributes: dict[str, object]) -> None:
                raise RuntimeError("start")

            def on_span(self, record: SpanRecord) -> None:
                raise RuntimeError("end")

        broken = Broken()
        add_sink(broken)
        try:
            with trace_span("resilient") as span:
                span["ok"] = True
        finally:
            remove_sink(broken)


@pytest.fixture
def otel() -> tuple[OtelSink, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    sink = setup_otel(processor=SimpleSpanProcessor(exporter))
    yield sink, exporter  # type: ignore[misc]
    remove_sink(sink)


class TestOtelExport:
    def test_spans_are_exported_with_hierarchy_and_attributes(
        self, otel: tuple[OtelSink, InMemorySpanExporter]
    ) -> None:
        _, exporter = otel

        with trace_span("pipeline", k=5):
            with trace_span("node.retrieve") as span:
                span["hits"] = 3
            with trace_span("llm.call", **{"gen_ai.request.model": "m"}) as llm_span:
                llm_span["gen_ai.usage.input_tokens"] = 120

        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert set(spans) == {"pipeline", "node.retrieve", "llm.call"}
        pipeline = spans["pipeline"]
        assert pipeline.parent is None
        assert pipeline.attributes is not None and pipeline.attributes["k"] == 5
        for child in ("node.retrieve", "llm.call"):
            assert spans[child].parent is not None
            assert spans[child].parent.span_id == pipeline.context.span_id  # type: ignore[union-attr]
            assert spans[child].context.trace_id == pipeline.context.trace_id
        assert spans["node.retrieve"].attributes["hits"] == 3  # type: ignore[index]
        assert spans["llm.call"].attributes["gen_ai.usage.input_tokens"] == 120  # type: ignore[index]

    def test_errors_mark_the_span_as_failed(
        self, otel: tuple[OtelSink, InMemorySpanExporter]
    ) -> None:
        _, exporter = otel

        with pytest.raises(ValueError), trace_span("failing"):
            raise ValueError("boom")

        (span,) = exporter.get_finished_spans()
        assert span.status.status_code is StatusCode.ERROR
        assert span.status.description == "ValueError"

    def test_non_primitive_attributes_are_stringified(
        self, otel: tuple[OtelSink, InMemorySpanExporter]
    ) -> None:
        _, exporter = otel

        with trace_span("demo", ids=["a", "b"], none=None):
            pass

        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes["ids"] == "['a', 'b']"
        assert "none" not in span.attributes


class TestOtelConfigHelpers:
    def test_parse_headers(self) -> None:
        assert parse_headers("Authorization=Basic abc, x-k=v=w") == {
            "Authorization": "Basic abc",
            "x-k": "v=w",
        }
        assert parse_headers(None) == {}

    def test_traces_endpoint_appends_path_once(self) -> None:
        assert (
            traces_endpoint("https://h/api/public/otel/")
            == "https://h/api/public/otel/v1/traces"
        )
        assert traces_endpoint("https://h/v1/traces") == "https://h/v1/traces"
