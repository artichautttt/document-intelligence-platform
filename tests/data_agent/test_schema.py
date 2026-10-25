"""La description du schéma donnée au modèle correspond à la base réelle."""

from document_intelligence.data_agent import DEFAULT_DB_PATH
from document_intelligence.data_agent.schema import SCHEMA, schema_as_text
from document_intelligence.data_agent.sql_guard import ALLOWED_TABLES, SafeSQLExecutor


def test_documented_tables_are_exactly_the_allow_list() -> None:
    assert {t.name for t in SCHEMA.tables} == ALLOWED_TABLES


def test_documented_columns_match_the_database() -> None:
    conn = SafeSQLExecutor(DEFAULT_DB_PATH).connect_read_only()
    try:
        for table in SCHEMA.tables:
            real = [
                (row[1], row[2])
                for row in conn.execute(f"PRAGMA table_info({table.name})")
            ]
            assert [(c.name, c.type) for c in table.columns] == real, table.name
    finally:
        conn.close()


def test_schema_text_lists_every_column() -> None:
    text = schema_as_text()
    for table in SCHEMA.tables:
        assert f"Table {table.name}" in text
        for column in table.columns:
            assert f"- {column.name} ({column.type})" in text
