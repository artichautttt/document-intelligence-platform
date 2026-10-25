"""Évaluation text-to-SQL : comparaison des valeurs, métriques, non-régression.

La suite de non-régression rejoue les 40 questions avec un client simulé qui
connaît les requêtes de référence : elle vérifie la mécanique (boucle d'outils,
garde-fous, exécution, format, métriques), pas la qualité d'un vrai modèle.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from document_intelligence.data_agent import DEFAULT_DB_PATH
from document_intelligence.data_agent.sql_guard import SafeSQLExecutor
from document_intelligence.evaluation.scripted_sql_client import ScriptedSQLClient
from document_intelligence.evaluation.sql_eval import (
    SCRIPTED_MODEL,
    run_sql_eval,
    summarize,
)
from document_intelligence.evaluation.sql_eval_cases import (
    SQLEvalCase,
    load_sql_eval,
    normalize_text,
    parse_number,
    value_matches,
)
from document_intelligence.orchestration.data_agent import DataAgent

CASES = load_sql_eval()


def case(answer_type: str, expected: Any, decimals: int | None = None) -> SQLEvalCase:
    return SQLEvalCase(
        id="x",
        difficulty="simple",
        provenance="test",
        question="?",
        answerable=True,
        answer_type=answer_type,  # type: ignore[arg-type]
        decimals=decimals,
        expected=expected,
        reference_sql="SELECT 1",
    )


class TestValueMatching:
    @pytest.mark.parametrize(
        "raw, number",
        [
            (61, 61.0),
            (27.12, 27.12),
            ("27,12", 27.12),
            ("3 330 486", 3330486.0),
            ("3,330,486", 3330486.0),
            ("73.4 %", 73.4),
            ("$4.49", 4.49),
            ("n/a", None),
            (True, None),
            (None, None),
        ],
    )
    def test_parse_number(self, raw: object, number: float | None) -> None:
        assert parse_number(raw) == number

    def test_integer_requires_exact_value(self) -> None:
        assert value_matches(case("integer", 120927), 120927)
        assert value_matches(case("integer", 120927), "120927")
        assert value_matches(case("integer", 120927), 120927.0)
        assert not value_matches(case("integer", 120927), 120928)

    def test_number_accepts_correct_rounding_only(self) -> None:
        assert value_matches(case("number", 27.12, 2), 27.12)
        assert value_matches(case("number", 27.12, 2), 27.1234)
        assert not value_matches(case("number", 27.12, 2), 27.5)
        # 0,5 % de la valeur attendue pour les grands montants
        assert value_matches(case("number", 90309683.0, 0), 90309683.4)
        assert not value_matches(case("number", 90309683.0, 0), 91_000_000)
        # demi-unité de la précision demandée pour les petites valeurs
        assert value_matches(case("number", 4.49, 2), 4.494)
        assert not value_matches(case("number", 4.49, 2), 4.6)

    def test_text_ignores_case_accents_and_trailing_punctuation(self) -> None:
        assert value_matches(case("text", "jeudi"), "Jeudi.")
        assert value_matches(
            case("text", "Mott Haven/Port Morris"), "mott haven/port morris"
        )
        assert not value_matches(case("text", "jeudi"), "vendredi")
        assert normalize_text("  Évolution ") == "evolution"

    def test_no_value_never_matches(self) -> None:
        assert not value_matches(case("integer", 1), None)


def _outcome(**overrides: Any) -> Any:
    from document_intelligence.evaluation.sql_eval import SQLCaseOutcome

    base: dict[str, Any] = {
        "id": "x",
        "difficulty": "simple",
        "question": "?",
        "answerable": True,
        "expected": 1,
        "outcome": "correct",
    }
    return SQLCaseOutcome(**{**base, **overrides})


def test_summarize_reports_each_difficulty_and_blocked_rules() -> None:
    outcomes = [
        _outcome(difficulty="simple", outcome="correct", latency_s=1.0, cost_usd=0.01),
        _outcome(difficulty="difficile", outcome="wrong_value", latency_s=3.0),
        _outcome(difficulty="difficile", outcome="correct", blocked_rules=["timeout"]),
        _outcome(
            difficulty="sans_reponse",
            answerable=False,
            expected=None,
            outcome="correct_refusal",
        ),
        _outcome(difficulty="intermediaire", outcome="false_refusal"),
    ]
    metrics = summarize(outcomes)
    assert metrics["execution_accuracy"] == 0.5
    assert metrics["accuracy_simple"] == 1.0
    assert metrics["accuracy_difficile"] == 0.5
    assert metrics["accuracy_intermediaire"] == 0.0
    assert metrics["correct_refusal_rate"] == 1.0
    assert metrics["false_refusal"] == 1 and metrics["wrong_value"] == 1
    assert metrics["blocked_queries"] == 1 and metrics["blocked_timeout"] == 1
    assert metrics["cost_usd"] == 0.01


def test_api_error_is_reported_not_raised() -> None:
    from document_intelligence.evaluation.sql_eval import evaluate_case
    from document_intelligence.orchestration.exceptions import LLMGenerationError

    class FailingAgent:
        def run(self, question: str) -> Any:
            raise LLMGenerationError("panne")

    outcome = evaluate_case(FailingAgent(), CASES[0])  # type: ignore[arg-type]
    assert outcome.outcome == "api_error" and outcome.error == "panne"


# --- suite de non-régression (client simulé, aucun appel payant) --------------


@pytest.fixture(scope="module")
def regression() -> tuple[list[Any], dict[str, float]]:
    agent = DataAgent(
        ScriptedSQLClient(CASES), SCRIPTED_MODEL, SafeSQLExecutor(DEFAULT_DB_PATH)
    )
    outcomes = run_sql_eval(agent, CASES, verbose=False)
    return outcomes, summarize(outcomes)


def test_regression_pipeline_answers_every_question(
    regression: tuple[list[Any], dict[str, float]],
) -> None:
    outcomes, metrics = regression
    failures = [(o.id, o.outcome) for o in outcomes if not o.correct]
    assert failures == []
    assert metrics["answerable_correct"] == 34 and metrics["correct_refusals"] == 6
    assert metrics["blocked_queries"] == 0 and metrics["sql_errors"] == 0
    assert metrics["cost_usd"] == 0.0


def test_regression_answers_cite_the_executed_reference_query(
    regression: tuple[list[Any], dict[str, float]],
) -> None:
    outcomes, _ = regression
    by_id = {c.id: c for c in CASES}
    for outcome in outcomes:
        if outcome.answerable:
            assert outcome.cited_sql == by_id[outcome.id].reference_sql
            assert outcome.answer.endswith(f"```sql\n{outcome.cited_sql}\n```")
        else:
            assert outcome.refused and outcome.cited_sql is None


class AttackingClient:
    """Modèle simulé qui tente des requêtes dangereuses, puis abandonne."""

    ATTACKS = [
        "DELETE FROM zones",
        "DROP TABLE od_monthly",
        "SELECT 1; DROP TABLE zones",
        "ATTACH DATABASE 'evil.db' AS evil",
        "PRAGMA writable_schema = ON",
        "SELECT sql FROM sqlite_master",
        "SELECT * FROM build_info",
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT max(x) FROM c",
        "SELECT load_extension('evil')",
        "SELECT COUNT(*) FROM od_monthly a, od_monthly b, od_monthly c",
    ]

    def __init__(self) -> None:
        self.messages = self
        self._step = 0

    def create(self, **kwargs: Any) -> SimpleNamespace:
        if self._step < len(self.ATTACKS):
            sql = self.ATTACKS[self._step]
            block = SimpleNamespace(
                type="tool_use", id=f"a{self._step}", name="run_sql", input={"sql": sql}
            )
        else:
            block = SimpleNamespace(
                type="tool_use",
                id="end",
                name="submit_answer",
                input={"answer": "abandon", "refused": True},
            )
        self._step += 1
        return SimpleNamespace(
            content=[block],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        )


def test_regression_every_attack_query_is_blocked_by_a_named_rule() -> None:
    executor = SafeSQLExecutor(DEFAULT_DB_PATH, timeout_s=0.3)
    before = DEFAULT_DB_PATH.read_bytes()
    agent = DataAgent(AttackingClient(), SCRIPTED_MODEL, executor, max_steps=12)
    result = agent.run("Supprime toutes les données.")

    assert [c.rule for c in result.sql_calls] == [
        "write_or_admin_keyword",
        "write_or_admin_keyword",
        "multiple_statements",
        "write_or_admin_keyword",
        "write_or_admin_keyword",
        "table_not_allowed",
        "table_not_allowed",
        "recursive_cte",
        "function_not_allowed",
        "timeout",
    ]
    assert result.refused and result.cited_sql is None
    assert DEFAULT_DB_PATH.read_bytes() == before
