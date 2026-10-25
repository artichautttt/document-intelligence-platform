"""Le jeu d'évaluation text-to-SQL est cohérent avec la base versionnée."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from document_intelligence.data_agent import DEFAULT_DB_PATH

DATASET = (
    Path(__file__).parents[2]
    / "src/document_intelligence/evaluation/golden/sql/taxi_sql_eval.jsonl"
)
ROWS: list[dict[str, Any]] = [
    json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()
]
ANSWERABLE = [r for r in ROWS if r["answerable"]]


@pytest.fixture(scope="module")
def conn() -> sqlite3.Connection:
    return sqlite3.connect(DEFAULT_DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)


def test_composition() -> None:
    levels = [r["difficulty"] for r in ROWS]
    assert len(ROWS) == 40
    assert len({r["id"] for r in ROWS}) == 40
    assert levels.count("difficile") >= 10
    assert levels.count("sans_reponse") == 6
    assert all(
        r["provenance"] == "écrit par l'auteur avec l'aide de Claude" for r in ROWS
    )


@pytest.mark.parametrize("row", ANSWERABLE, ids=[r["id"] for r in ANSWERABLE])
def test_reference_sql_gives_expected(
    conn: sqlite3.Connection, row: dict[str, Any]
) -> None:
    rows = conn.execute(row["reference_sql"]).fetchall()
    assert len(rows) == 1 and len(rows[0]) == 1
    value = rows[0][0]
    if row["answer_type"] == "integer":
        value = int(value)
    assert value == row["expected"]


@pytest.mark.parametrize(
    "row",
    [r for r in ANSWERABLE if r["tie_check_sql"]],
    ids=[r["id"] for r in ANSWERABLE if r["tie_check_sql"]],
)
def test_rankings_have_no_tie(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    first, second = (r[0] for r in conn.execute(row["tie_check_sql"]).fetchall())
    assert first != second


def test_unanswerable_rows_have_no_reference() -> None:
    for row in ROWS:
        if not row["answerable"]:
            assert row["reference_sql"] is None and row["expected"] is None
            assert row["why_unanswerable"]
