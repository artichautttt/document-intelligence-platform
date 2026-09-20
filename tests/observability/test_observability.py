"""Tests du tracing et du registre de métriques."""

import pytest

from document_intelligence.core import config
from document_intelligence.observability import (
    MetricsRegistry,
    SpanRecord,
    SpanSink,
    add_sink,
    registry,
    remove_sink,
    trace_span,
)


class RecordingSink(SpanSink):
    def __init__(self) -> None:
        self.records: list[SpanRecord] = []

    def on_span(self, record: SpanRecord) -> None:
        self.records.append(record)


class TestTraceSpan:
    def test_records_duration_and_attributes(self) -> None:
        sink = RecordingSink()
        add_sink(sink)
        try:
            with trace_span("demo", k=3) as span:
                span["hits"] = 2
        finally:
            remove_sink(sink)

        (record,) = sink.records
        assert record.name == "demo"
        assert record.attributes == {"k": 3, "hits": 2}
        assert record.duration_ms >= 0
        assert record.error is None

    def test_records_error_and_reraises(self) -> None:
        sink = RecordingSink()
        add_sink(sink)
        try:
            with pytest.raises(ValueError), trace_span("failing"):
                raise ValueError("boom")
        finally:
            remove_sink(sink)

        assert sink.records[0].error == "ValueError"


class TestMetricsRegistry:
    def test_aggregates_spans_with_percentiles_and_errors(self) -> None:
        reg = MetricsRegistry()
        for duration in (10.0, 20.0, 30.0, 40.0):
            reg.on_span(SpanRecord(name="node.retrieve", duration_ms=duration))
        reg.on_span(SpanRecord(name="node.retrieve", duration_ms=50.0, error="X"))

        stats = reg.snapshot()["spans"]["node.retrieve"]
        assert stats["count"] == 5
        assert stats["errors"] == 1
        assert stats["latency_ms_avg"] == 30.0
        assert stats["latency_ms_p50"] == 30.0
        assert stats["latency_ms_p95"] == 50.0

    def test_llm_usage_and_cost_use_configured_pricing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config.settings, "llm_input_price_per_mtok", 3.0)
        monkeypatch.setattr(config.settings, "llm_output_price_per_mtok", 15.0)
        reg = MetricsRegistry()
        reg.record_llm_call(
            input_tokens=1_000_000, output_tokens=200_000, latency_ms=120
        )

        llm = reg.snapshot()["llm"]
        assert llm["calls"] == 1
        assert llm["input_tokens"] == 1_000_000
        assert llm["estimated_cost_usd"] == 6.0  # 3.0 + 0.2 * 15.0
        assert llm["pricing_configured"] is True

    def test_cost_is_zero_and_flagged_when_pricing_not_configured(self) -> None:
        reg = MetricsRegistry()
        reg.record_llm_call(input_tokens=500, output_tokens=500, latency_ms=10)

        llm = reg.snapshot()["llm"]
        assert llm["estimated_cost_usd"] == 0.0
        assert llm["pricing_configured"] is False

    def test_global_registry_collects_spans_from_trace_span(self) -> None:
        registry.reset()
        with trace_span("global.demo"):
            pass

        assert registry.snapshot()["spans"]["global.demo"]["count"] == 1
        registry.reset()
