"""Persistance SQLite des métriques, pour survivre aux redémarrages du processus."""

import sqlite3
import threading
import time
from pathlib import Path

from document_intelligence.core.logging import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS span_events (
    ts REAL NOT NULL,
    name TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_span_events_name ON span_events(name, ts);
CREATE TABLE IF NOT EXISTS llm_events (
    ts REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_events_ts ON llm_events(ts);
"""


class SqliteMetricsStore:
    """Journal d'événements de métriques (spans et appels LLM) dans un fichier SQLite.

    Les écritures ne lèvent jamais : une erreur disque est journalisée, la
    collecte en mémoire continue. Les événements plus anciens que
    `retention_days` sont purgés à l'ouverture.
    """

    def __init__(self, path: str | Path, retention_days: int = 30) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        cutoff = time.time() - retention_days * 86400
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM span_events WHERE ts < ?", (cutoff,))
            self._conn.execute("DELETE FROM llm_events WHERE ts < ?", (cutoff,))

    def append_span(self, name: str, duration_ms: float, error: str | None) -> None:
        self._write(
            "INSERT INTO span_events VALUES (?, ?, ?, ?)",
            (time.time(), name, duration_ms, error),
        )

    def append_llm_call(
        self, input_tokens: int, output_tokens: int, latency_ms: float
    ) -> None:
        self._write(
            "INSERT INTO llm_events VALUES (?, ?, ?, ?)",
            (time.time(), input_tokens, output_tokens, latency_ms),
        )

    def _write(self, sql: str, params: tuple[object, ...]) -> None:
        try:
            with self._lock, self._conn:
                self._conn.execute(sql, params)
        except sqlite3.Error:
            logger.warning("metrics.store_write_failed", exc_info=True)

    def span_totals(self) -> dict[str, tuple[int, int]]:
        """Retourne `{nom: (nombre de spans, nombre d'erreurs)}`."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, COUNT(*), COUNT(error) FROM span_events GROUP BY name"
            ).fetchall()
        return {name: (count, errors) for name, count, errors in rows}

    def recent_durations(self, name: str, limit: int) -> list[float]:
        """Durées des `limit` derniers spans de ce nom, du plus ancien au plus récent."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT duration_ms FROM span_events WHERE name = ? "
                "ORDER BY ts DESC, rowid DESC LIMIT ?",
                (name, limit),
            ).fetchall()
        return [row[0] for row in reversed(rows)]

    def llm_totals(self) -> tuple[int, int, int]:
        """Retourne `(appels, tokens d'entrée, tokens de sortie)`."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), "
                "COALESCE(SUM(output_tokens), 0) FROM llm_events"
            ).fetchone()
        return int(row[0]), int(row[1]), int(row[2])

    def recent_llm_latencies(self, limit: int) -> list[float]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT latency_ms FROM llm_events ORDER BY ts DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [row[0] for row in reversed(rows)]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
