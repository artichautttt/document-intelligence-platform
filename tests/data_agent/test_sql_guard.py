"""Garde-fous SQL : chaque règle, et chaque couche indépendamment des autres."""

import sqlite3
from pathlib import Path

import pytest

from document_intelligence.data_agent import DEFAULT_DB_PATH
from document_intelligence.data_agent.sql_guard import (
    ALLOWED_TABLES,
    MAX_SQL_CHARS,
    SafeSQLExecutor,
    SQLExecutionError,
    SQLGuardError,
    check_sql,
)


@pytest.fixture(scope="module")
def executor() -> SafeSQLExecutor:
    return SafeSQLExecutor(DEFAULT_DB_PATH)


def _rule(excinfo: pytest.ExceptionInfo[SQLGuardError]) -> str:
    return excinfo.value.rule


# --- couche 1 : contrôles lexicaux -------------------------------------------


class TestLexicalChecks:
    @pytest.mark.parametrize("sql", ["", "   ", "\n\t"])
    def test_empty_query(self, sql: str) -> None:
        with pytest.raises(SQLGuardError) as exc:
            check_sql(sql)
        assert _rule(exc) == "empty_query"

    def test_too_long(self) -> None:
        with pytest.raises(SQLGuardError) as exc:
            check_sql("SELECT 1 " + " " * MAX_SQL_CHARS)
        assert _rule(exc) == "too_long"

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1; SELECT 2",
            "SELECT * FROM zones; DROP TABLE zones",
            "SELECT 1;SELECT 2;",
        ],
    )
    def test_multiple_statements(self, sql: str) -> None:
        with pytest.raises(SQLGuardError) as exc:
            check_sql(sql)
        assert _rule(exc) == "multiple_statements"

    @pytest.mark.parametrize(
        "sql, keyword",
        [
            ("INSERT INTO zones VALUES (999, 'x', 'y', 'z')", "insert"),
            ("UPDATE zones SET zone = 'x'", "update"),
            ("DELETE FROM zones", "delete"),
            ("DROP TABLE zones", "drop"),
            ("ALTER TABLE zones ADD COLUMN x", "alter"),
            ("CREATE TABLE t (x)", "create"),
            ("ATTACH DATABASE 'x.db' AS x", "attach"),
            ("PRAGMA table_info(zones)", "pragma"),
            ("VACUUM", "vacuum"),
            ("WITH x AS (SELECT 1) DELETE FROM zones", "delete"),
        ],
    )
    def test_write_or_admin_keyword_is_refused_explicitly(
        self, sql: str, keyword: str
    ) -> None:
        with pytest.raises(SQLGuardError) as exc:
            check_sql(sql)
        assert _rule(exc) == "write_or_admin_keyword"
        assert keyword in str(exc.value)
        assert "lecture seule" in str(exc.value)

    @pytest.mark.parametrize(
        "sql", ["EXPLAIN SELECT 1", "VALUES (1)", "REPLACE INTO zones VALUES (1)"]
    )
    def test_not_select(self, sql: str) -> None:
        with pytest.raises(SQLGuardError) as exc:
            check_sql(sql)
        assert _rule(exc) == "not_select"

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT ';' AS x",
            "SELECT 'DROP TABLE zones' AS x",
            "SELECT 1 -- ; DELETE FROM zones",
            "SELECT 1 /* ; drop */",
            "  select count(*) from zones;  ",
            "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
            'SELECT "zone" FROM zones',
        ],
    )
    def test_accepts_selects_with_literals_and_comments(self, sql: str) -> None:
        assert check_sql(sql)

    def test_recursive_cte(self) -> None:
        with pytest.raises(SQLGuardError) as exc:
            check_sql(
                "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT 1"
            )
        assert _rule(exc) == "recursive_cte"

    def test_strips_final_semicolon(self) -> None:
        assert check_sql("SELECT 1 ;  ") == "SELECT 1"


# --- couche 2 : connexion en lecture seule -----------------------------------


def test_read_only_connection_rejects_writes_without_authorizer(
    executor: SafeSQLExecutor,
) -> None:
    conn = executor.connect_read_only()
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            conn.execute("DELETE FROM zones")
    finally:
        conn.close()


# --- couche 3 : autorisateur du moteur (contrôles lexicaux contournés) --------


class TestEngineAuthorizer:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT name FROM sqlite_master",
            "SELECT count(*) FROM sqlite_master",
            "SELECT * FROM build_info",
            "SELECT z.zone FROM zones z JOIN build_info b ON 1 = 1",
        ],
    )
    def test_table_outside_allow_list(
        self, executor: SafeSQLExecutor, sql: str
    ) -> None:
        with pytest.raises(SQLGuardError) as exc:
            executor.run_engine_guarded(sql)
        assert _rule(exc) == "table_not_allowed"

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT load_extension('x')",
            "SELECT randomblob(1000000000)",
            "SELECT zeroblob(1000000000)",
            "SELECT random()",
        ],
    )
    def test_function_outside_allow_list(
        self, executor: SafeSQLExecutor, sql: str
    ) -> None:
        with pytest.raises(SQLGuardError) as exc:
            executor.run_engine_guarded(sql)
        assert _rule(exc) == "function_not_allowed"

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM zones",
            "INSERT INTO zones VALUES (999, 'x', 'y', 'z')",
            "UPDATE zones SET zone = 'x'",
            "DROP TABLE zones",
            "CREATE TEMP TABLE t (x)",
            "ATTACH DATABASE ':memory:' AS other",
            "PRAGMA table_info(zones)",
        ],
    )
    def test_writes_and_admin_denied_by_the_engine(
        self, executor: SafeSQLExecutor, sql: str
    ) -> None:
        with pytest.raises(SQLGuardError) as exc:
            executor.run_engine_guarded(sql)
        assert _rule(exc) == "operation_not_allowed"

    def test_recursive_cte_denied(self, executor: SafeSQLExecutor) -> None:
        # SQLite signale la lecture de la CTE (nom hors liste blanche) avant la
        # récursion : refus sous la règle table_not_allowed.
        sql = (
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
            "SELECT count(*) FROM c"
        )
        with pytest.raises(SQLGuardError) as exc:
            executor.run_engine_guarded(sql)
        assert _rule(exc) == "table_not_allowed"

    def test_recursive_cte_named_like_an_allowed_table(
        self, executor: SafeSQLExecutor
    ) -> None:
        sql = (
            "WITH RECURSIVE zones(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM zones) "
            "SELECT count(*) FROM zones"
        )
        with pytest.raises(SQLGuardError) as exc:
            executor.run_engine_guarded(sql)
        assert _rule(exc) == "recursive_cte"

    def test_window_functions_ctes_and_joins_are_allowed(
        self, executor: SafeSQLExecutor
    ) -> None:
        sql = """
            WITH daily AS (
                SELECT trip_date, SUM(trip_count) AS n FROM hourly_trips GROUP BY trip_date
            )
            SELECT trip_date, n,
                   RANK() OVER (ORDER BY n DESC) AS rnk,
                   n - LAG(n) OVER (ORDER BY trip_date) AS delta,
                   SUM(n) OVER (ORDER BY trip_date ROWS UNBOUNDED PRECEDING) AS cum,
                   strftime('%w', trip_date) AS weekday,
                   ROUND(100.0 * n / SUM(n) OVER (), 2) AS pct
            FROM daily ORDER BY trip_date LIMIT 3
        """
        result = executor.run_engine_guarded(sql)
        assert result.columns == [
            "trip_date",
            "n",
            "rnk",
            "delta",
            "cum",
            "weekday",
            "pct",
        ]
        assert result.rows[0][0] == "2025-01-01"
        assert result.rows[0][3] is None  # pas de veille pour le 1er janvier

    def test_every_allowed_table_is_readable(self, executor: SafeSQLExecutor) -> None:
        for table in sorted(ALLOWED_TABLES):
            assert (
                executor.run_engine_guarded(f"SELECT COUNT(*) FROM {table}").rows[0][0]
                > 0
            )


# --- couche 4 : limites de lignes et de temps --------------------------------


def test_row_limit_truncates_and_says_so() -> None:
    small = SafeSQLExecutor(DEFAULT_DB_PATH, max_rows=5)
    result = small.execute("SELECT location_id FROM zones ORDER BY location_id")
    assert result.row_count == 5
    assert result.truncated is True
    assert result.rows == [[1], [2], [3], [4], [5]]


def test_result_within_limit_is_not_truncated(executor: SafeSQLExecutor) -> None:
    result = executor.execute("SELECT label FROM payment_types ORDER BY payment_type")
    assert result.row_count == 7 and result.truncated is False


def test_timeout_interrupts_a_long_query() -> None:
    fast = SafeSQLExecutor(DEFAULT_DB_PATH, timeout_s=0.2)
    with pytest.raises(SQLGuardError) as exc:
        fast.execute("SELECT COUNT(*) FROM od_monthly a, od_monthly b")
    assert _rule(exc) == "timeout"
    assert "0.2 s" in str(exc.value)


# --- comportement général -----------------------------------------------------


def test_invalid_sql_is_an_execution_error_not_a_guard(
    executor: SafeSQLExecutor,
) -> None:
    with pytest.raises(SQLExecutionError, match="no such column"):
        executor.execute("SELECT nope FROM zones")


def test_execute_runs_a_valid_select(executor: SafeSQLExecutor) -> None:
    result = executor.execute(
        "SELECT COUNT(*) AS n FROM zones WHERE borough = 'Brooklyn';"
    )
    assert result.columns == ["n"]
    assert result.rows == [[61]]
    assert result.sql == "SELECT COUNT(*) AS n FROM zones WHERE borough = 'Brooklyn'"


def test_missing_database_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SafeSQLExecutor(tmp_path / "absent.sqlite")


def test_database_is_unchanged_after_attacks(executor: SafeSQLExecutor) -> None:
    before = DEFAULT_DB_PATH.read_bytes()
    for sql in ("DELETE FROM zones", "DROP TABLE zones", "UPDATE zones SET zone = 'x'"):
        with pytest.raises(SQLGuardError):
            executor.execute(sql)
        with pytest.raises(SQLGuardError):
            executor.run_engine_guarded(sql)
    assert DEFAULT_DB_PATH.read_bytes() == before
