"""Exécution SQL en lecture seule, protégée par plusieurs couches indépendantes.

Une requête doit franchir, dans l'ordre :

1. **contrôles lexicaux** (`check_sql`) : requête non vide, longueur bornée,
   une seule instruction, première instruction `SELECT` ou `WITH`, aucun mot-clé
   d'écriture ou d'administration, pas de CTE récursive. Commentaires et littéraux sont retirés
   avant l'analyse (un « ; » dans une chaîne n'est pas une seconde instruction).
   Ces contrôles produisent un refus explicite et lisible ;
2. **connexion en lecture seule** (`mode=ro`, `PRAGMA query_only`) ;
3. **autorisateur du moteur** (`sqlite3.Connection.set_authorizer`) : SQLite
   consulte cette fonction pour chaque opération au moment de la compilation.
   Seuls `SELECT`, la lecture des tables de `ALLOWED_TABLES` et les fonctions
   de `ALLOWED_FUNCTIONS` sont acceptés ; tout le reste (écriture, `ATTACH`,
   `PRAGMA`, tables système, CTE récursive...) est refusé, même si la requête a
   trompé les contrôles lexicaux ;
4. **limite de temps** (gestionnaire de progression) et **limite de lignes**
   (résultat tronqué, signalé par `truncated`).

Chaque refus lève `SQLGuardError` avec le nom de la règle (`rule`).
"""

import re
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from document_intelligence.core.logging import get_logger
from document_intelligence.data_agent import DEFAULT_DB_PATH

logger = get_logger(__name__)

ALLOWED_TABLES = frozenset(
    {
        "zones",
        "payment_types",
        "daily_zone_trips",
        "hourly_trips",
        "payment_daily",
        "od_monthly",
    }
)

# Fonctions usuelles : agrégats, fenêtres, texte, dates, math. Absentes, donc
# refusées : load_extension, randomblob / zeroblob (allocation), random, etc.
ALLOWED_FUNCTIONS = frozenset(
    {
        # agrégats
        "count", "sum", "total", "avg", "min", "max", "group_concat",
        # fenêtres
        "row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile",
        "lag", "lead", "first_value", "last_value", "nth_value",
        # scalaires
        "abs", "round", "coalesce", "ifnull", "nullif", "iif", "typeof",
        "lower", "upper", "length", "substr", "substring", "trim", "ltrim",
        "rtrim", "replace", "instr", "printf", "format", "like", "glob",
        # dates
        "date", "time", "datetime", "julianday", "strftime", "unixepoch",
        # math
        "sqrt", "power", "pow", "exp", "ln", "log", "log10", "floor", "ceil",
        "ceiling", "mod", "sign",
    }
)  # fmt: skip

WRITE_OR_ADMIN_KEYWORDS = frozenset(
    {
        "insert", "update", "delete", "drop", "alter", "create", "attach",
        "detach", "pragma", "vacuum", "reindex", "analyze", "begin", "commit",
        "rollback", "savepoint", "release", "upsert",
    }
)  # fmt: skip

DEFAULT_MAX_ROWS = 200
DEFAULT_TIMEOUT_S = 2.0
MAX_SQL_CHARS = 4000
_PROGRESS_STEPS = 10_000

_LITERAL_OR_COMMENT = re.compile(
    r"'(?:[^']|'')*'"  # chaîne
    r'|"(?:[^"]|"")*"'  # identifiant entre guillemets
    r"|`[^`]*`|\[[^\]]*\]"  # identifiants `x` ou [x]
    r"|--[^\n]*"  # commentaire ligne
    r"|/\*.*?(?:\*/|$)",  # commentaire bloc (éventuellement non fermé)
    re.DOTALL,
)
_WORD = re.compile(r"[a-z_]+")


class SQLGuardError(ValueError):
    """Requête refusée par un garde-fou ; `rule` nomme la règle déclenchée."""

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(message)
        self.rule = rule


class SQLExecutionError(ValueError):
    """Requête autorisée mais invalide pour SQLite (syntaxe, colonne inconnue...)."""


class SQLResult(BaseModel):
    """Résultat d'une requête exécutée."""

    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    duration_ms: float


def _strip_literals_and_comments(sql: str) -> str:
    """Remplace littéraux, identifiants cités et commentaires par un espace."""

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        # Un identifiant cité reste un identifiant : on garde un mot neutre.
        return " _ident_ " if token[0] in '"`[' else " "

    return _LITERAL_OR_COMMENT.sub(replace, sql)


def check_sql(sql: str) -> str:
    """Contrôles lexicaux ; renvoie la requête nettoyée du « ; » final.

    Raises:
        SQLGuardError: requête vide, trop longue, multiple, non `SELECT`, ou
            contenant un mot-clé d'écriture / d'administration.
    """
    if not sql.strip():
        raise SQLGuardError("empty_query", "Requête SQL vide.")
    if len(sql) > MAX_SQL_CHARS:
        raise SQLGuardError(
            "too_long",
            f"Requête trop longue ({len(sql)} > {MAX_SQL_CHARS} caractères).",
        )
    code = _strip_literals_and_comments(sql).strip().lower()
    statements = [part for part in code.split(";") if part.strip()]
    if len(statements) > 1:
        raise SQLGuardError(
            "multiple_statements", "Une seule instruction SQL est autorisée par appel."
        )
    words = _WORD.findall(code)
    forbidden = sorted(WRITE_OR_ADMIN_KEYWORDS.intersection(words))
    if forbidden:
        raise SQLGuardError(
            "write_or_admin_keyword",
            "Requête refusée : base en lecture seule, mot-clé interdit "
            f"({', '.join(forbidden)}). Seules les requêtes SELECT sont autorisées.",
        )
    if "recursive" in words:
        raise SQLGuardError(
            "recursive_cte",
            "Requête refusée : les CTE récursives ne sont pas autorisées.",
        )
    if not words or words[0] not in ("select", "with"):
        raise SQLGuardError(
            "not_select",
            "Requête refusée : seules les requêtes SELECT (ou WITH ... SELECT) "
            "sont autorisées.",
        )
    return sql.strip().rstrip(";").strip()


class SafeSQLExecutor:
    """Exécute des requêtes `SELECT` sur une base SQLite, derrière les garde-fous.

    Une connexion est ouverte par requête : l'exécuteur peut être partagé
    entre threads (outils MCP exécutés hors de la boucle d'événements).
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        *,
        max_rows: int = DEFAULT_MAX_ROWS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        allowed_tables: frozenset[str] = ALLOWED_TABLES,
    ) -> None:
        if not db_path.is_file():
            raise FileNotFoundError(f"Base SQLite introuvable : {db_path}")
        self.db_path = db_path
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self.allowed_tables = allowed_tables

    def connect_read_only(self) -> sqlite3.Connection:
        """Connexion en lecture seule, sans autorisateur (couche 2 seule)."""
        conn = sqlite3.connect(
            self.db_path.resolve().as_uri() + "?mode=ro",
            uri=True,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.execute("PRAGMA query_only = ON")
        return conn

    @contextmanager
    def _guarded_connection(self, denials: list[str]) -> Iterator[sqlite3.Connection]:
        conn = self.connect_read_only()
        conn.set_authorizer(self._authorizer(denials))
        try:
            yield conn
        finally:
            conn.close()

    def _authorizer(self, denials: list[str]) -> Any:
        allowed_tables = self.allowed_tables

        def authorize(
            action: int,
            arg1: str | None,
            arg2: str | None,
            _db: str | None,
            _trigger: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ:
                if arg1 in allowed_tables:
                    return sqlite3.SQLITE_OK
                denials.append(f"table_not_allowed:{arg1}")
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION:
                if (arg2 or "").lower() in ALLOWED_FUNCTIONS:
                    return sqlite3.SQLITE_OK
                denials.append(f"function_not_allowed:{arg2}")
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_RECURSIVE:
                denials.append("recursive_cte")
                return sqlite3.SQLITE_DENY
            denials.append(f"operation_not_allowed:{action}")
            return sqlite3.SQLITE_DENY

        return authorize

    def execute(self, sql: str) -> SQLResult:
        """Contrôles lexicaux puis exécution protégée (toutes les couches)."""
        try:
            cleaned = check_sql(sql)
        except SQLGuardError as exc:
            logger.warning("data_agent.sql_blocked", rule=exc.rule, sql=sql[:500])
            raise
        return self.run_engine_guarded(cleaned)

    def run_engine_guarded(self, sql: str) -> SQLResult:
        """Exécution protégée par les couches 2 à 4 seules (sans contrôle lexical).

        Séparée de `execute` pour tester chaque couche indépendamment.
        """
        denials: list[str] = []
        started = time.perf_counter()
        deadline = started + self.timeout_s
        with self._guarded_connection(denials) as conn:
            conn.set_progress_handler(
                lambda: 1 if time.perf_counter() > deadline else 0, _PROGRESS_STEPS
            )
            try:
                cursor = conn.execute(sql)
                fetched = cursor.fetchmany(self.max_rows + 1)
            except sqlite3.Warning as exc:  # ex. plusieurs instructions
                raise SQLGuardError("multiple_statements", str(exc)) from exc
            except sqlite3.Error as exc:
                raise self._translate(exc, denials, sql) from exc
            columns = [d[0] for d in cursor.description or []]
        truncated = len(fetched) > self.max_rows
        rows = [[_jsonable(v) for v in row] for row in fetched[: self.max_rows]]
        return SQLResult(
            sql=sql,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def _translate(
        self, exc: sqlite3.Error, denials: list[str], sql: str
    ) -> SQLGuardError | SQLExecutionError:
        message = str(exc)
        if denials:
            rule = denials[0].split(":", 1)[0]
            detail = denials[0].split(":", 1)[1] if ":" in denials[0] else ""
            text = {
                "table_not_allowed": f"table non autorisée ({detail}). Tables "
                f"autorisées : {', '.join(sorted(self.allowed_tables))}.",
                "function_not_allowed": f"fonction SQL non autorisée ({detail}).",
                "recursive_cte": "les CTE récursives ne sont pas autorisées.",
                "operation_not_allowed": "opération non autorisée (lecture seule).",
            }[rule]
            logger.warning("data_agent.sql_blocked", rule=rule, sql=sql[:500])
            return SQLGuardError(rule, f"Requête refusée : {text}")
        if "interrupted" in message:
            logger.warning("data_agent.sql_blocked", rule="timeout", sql=sql[:500])
            return SQLGuardError(
                "timeout",
                f"Requête interrompue : durée supérieure à {self.timeout_s} s.",
            )
        if "readonly" in message or "read-only" in message:
            return SQLGuardError("read_only", f"Requête refusée : {message}")
        return SQLExecutionError(f"Erreur SQLite : {message}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    return value
