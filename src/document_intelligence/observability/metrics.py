"""Registre de métriques : latence par étape, tokens et coût LLM.

Les agrégats vivent en mémoire ; si un `SqliteMetricsStore` est attaché, chaque
événement y est aussi journalisé et l'état est restauré au redémarrage.
"""

import math
import threading
from collections import defaultdict, deque
from typing import Any

from document_intelligence.core.config import settings
from document_intelligence.observability.store import SqliteMetricsStore
from document_intelligence.observability.tracing import SpanRecord, SpanSink

_MAX_SAMPLES = 1000


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, math.ceil(q * len(sorted_values)) - 1)
    return sorted_values[index]


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Coût estimé d'un appel, d'après les tarifs configurés (0.0 si non configurés)."""
    return (
        input_tokens * settings.llm_input_price_per_mtok
        + output_tokens * settings.llm_output_price_per_mtok
    ) / 1_000_000


class MetricsRegistry(SpanSink):
    """Agrège les spans et les appels LLM ; sûr en accès concurrent."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: SqliteMetricsStore | None = None
        self.reset()

    def reset(self) -> None:
        """Remet à zéro les agrégats en mémoire (le journal persistant est conservé)."""
        with self._lock:
            self._durations: dict[str, deque[float]] = defaultdict(
                lambda: deque(maxlen=_MAX_SAMPLES)
            )
            self._counts: dict[str, int] = defaultdict(int)
            self._errors: dict[str, int] = defaultdict(int)
            self._llm_calls = 0
            self._input_tokens = 0
            self._output_tokens = 0
            self._llm_latencies: deque[float] = deque(maxlen=_MAX_SAMPLES)

    def attach_store(self, store: SqliteMetricsStore) -> None:
        """Restaure l'historique depuis `store`, puis y journalise les nouveaux événements."""
        self.reset()
        with self._lock:
            for name, (count, errors) in store.span_totals().items():
                self._counts[name] = count
                self._errors[name] = errors
                self._durations[name].extend(store.recent_durations(name, _MAX_SAMPLES))
            self._llm_calls, self._input_tokens, self._output_tokens = (
                store.llm_totals()
            )
            self._llm_latencies.extend(store.recent_llm_latencies(_MAX_SAMPLES))
            self._store = store

    def detach_store(self) -> None:
        """Cesse de journaliser (ne ferme pas le store)."""
        with self._lock:
            self._store = None

    def on_span(self, record: SpanRecord) -> None:
        with self._lock:
            self._counts[record.name] += 1
            self._durations[record.name].append(record.duration_ms)
            if record.error is not None:
                self._errors[record.name] += 1
            store = self._store
        if store is not None:
            store.append_span(record.name, record.duration_ms, record.error)

    def record_llm_call(
        self, input_tokens: int, output_tokens: int, latency_ms: float
    ) -> None:
        """Comptabilise un appel LLM (tokens consommés et latence)."""
        with self._lock:
            self._llm_calls += 1
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._llm_latencies.append(latency_ms)
            store = self._store
        if store is not None:
            store.append_llm_call(input_tokens, output_tokens, latency_ms)

    def snapshot(self) -> dict[str, Any]:
        """Vue JSON-sérialisable de l'état courant des métriques."""
        with self._lock:
            spans = {}
            for name, count in self._counts.items():
                durations = sorted(self._durations[name])
                spans[name] = {
                    "count": count,
                    "errors": self._errors[name],
                    "latency_ms_avg": round(sum(durations) / len(durations), 2),
                    "latency_ms_p50": round(_percentile(durations, 0.50), 2),
                    "latency_ms_p95": round(_percentile(durations, 0.95), 2),
                }
            llm_latencies = sorted(self._llm_latencies)
            return {
                "spans": spans,
                "llm": {
                    "calls": self._llm_calls,
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                    "latency_ms_p50": round(_percentile(llm_latencies, 0.50), 2),
                    "latency_ms_p95": round(_percentile(llm_latencies, 0.95), 2),
                    "estimated_cost_usd": round(
                        estimate_cost_usd(self._input_tokens, self._output_tokens), 6
                    ),
                    "pricing_configured": (
                        settings.llm_input_price_per_mtok > 0
                        or settings.llm_output_price_per_mtok > 0
                    ),
                },
                "persistent": self._store is not None,
            }


registry = MetricsRegistry()
