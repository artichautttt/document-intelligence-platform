"""Évaluation de l'agent de données text-to-SQL, par exécution.

    # Non-régression, sans appel payant (client simulé, requêtes de référence) :
    uv run python -m document_intelligence.evaluation.sql_eval --scripted
    # Mesure réelle (claude-sonnet-5 par défaut), suivie dans MLflow :
    uv run python -m document_intelligence.evaluation.sql_eval --output rapport_sql.json

Pour chaque question : l'agent écrit et exécute son SQL, puis rend une valeur.
Elle est comparée au résultat de la requête de référence (`value_matches`). Les
questions sans réponse possible doivent être refusées.

Le mode `--scripted` ne mesure pas la qualité d'un modèle : le client simulé
connaît les requêtes de référence. Il vérifie la mécanique de bout en bout.
"""

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from document_intelligence.core.config import settings
from document_intelligence.core.logging import configure_logging
from document_intelligence.data_agent.sql_guard import SafeSQLExecutor
from document_intelligence.evaluation.scripted_sql_client import ScriptedSQLClient
from document_intelligence.evaluation.sql_eval_cases import (
    DIFFICULTIES,
    SQLEvalCase,
    load_sql_eval,
    value_matches,
)
from document_intelligence.evaluation.tracking import log_data_agent_run
from document_intelligence.orchestration.cost import CostLedger
from document_intelligence.orchestration.data_agent import DataAgent, DataAnswer
from document_intelligence.orchestration.exceptions import LLMGenerationError
from document_intelligence.orchestration.tiering import LLMRole, model_for

SCRIPTED_MODEL = "scripted-reference-sql"


@dataclass
class SQLCaseOutcome:
    """Résultat d'une question, avec tout ce qu'il faut pour l'analyser ensuite."""

    id: str
    difficulty: str
    question: str
    answerable: bool
    expected: Any
    outcome: str  # correct | wrong_value | false_refusal | correct_refusal | missed_refusal | no_submission | api_error
    value: Any = None
    refused: bool = False
    answer: str = ""
    cited_sql: str | None = None
    sql_calls: list[dict[str, Any]] = field(default_factory=list)
    blocked_rules: list[str] = field(default_factory=list)
    steps: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.outcome in ("correct", "correct_refusal")


def classify(case: SQLEvalCase, result: DataAnswer) -> str:
    if not result.submitted:
        return "no_submission"
    if not case.answerable:
        return "correct_refusal" if result.refused else "missed_refusal"
    if result.refused:
        return "false_refusal"
    return "correct" if value_matches(case, result.value) else "wrong_value"


def evaluate_case(agent: DataAgent, case: SQLEvalCase) -> SQLCaseOutcome:
    outcome = SQLCaseOutcome(
        id=case.id,
        difficulty=case.difficulty,
        question=case.question,
        answerable=case.answerable,
        expected=case.expected,
        outcome="api_error",
    )
    try:
        result = agent.run(case.question)
    except LLMGenerationError as exc:
        outcome.error = str(exc)
        return outcome
    outcome.outcome = classify(case, result)
    outcome.value = result.value
    outcome.refused = result.refused
    outcome.answer = result.formatted()
    outcome.cited_sql = result.cited_sql
    outcome.sql_calls = [c.model_dump() for c in result.sql_calls]
    outcome.blocked_rules = [c.rule for c in result.blocked_calls if c.rule]
    outcome.steps = result.steps
    outcome.latency_s = result.latency_s
    outcome.cost_usd = result.cost_usd
    outcome.input_tokens = result.input_tokens
    outcome.output_tokens = result.output_tokens
    return outcome


def run_sql_eval(
    agent: DataAgent, cases: list[SQLEvalCase], verbose: bool = True
) -> list[SQLCaseOutcome]:
    outcomes = []
    for index, case in enumerate(cases, 1):
        outcome = evaluate_case(agent, case)
        outcomes.append(outcome)
        if verbose:
            print(
                f"[{index}/{len(cases)}] {case.id} {outcome.outcome:<15} "
                f"{outcome.latency_s:5.1f}s  {case.question[:60]}"
            )
    return outcomes


def summarize(outcomes: list[SQLCaseOutcome]) -> dict[str, float]:
    """Métriques agrégées ; les taux sont des fractions (0 à 1)."""
    metrics: dict[str, float] = {"questions": len(outcomes)}
    answerable = [o for o in outcomes if o.answerable]
    unanswerable = [o for o in outcomes if not o.answerable]
    if answerable:
        metrics["execution_accuracy"] = _rate(answerable, "correct")
        metrics["answerable_correct"] = _count(answerable, "correct")
        metrics["answerable_total"] = len(answerable)
    for level in DIFFICULTIES[:3]:
        group = [o for o in answerable if o.difficulty == level]
        if group:
            metrics[f"accuracy_{level}"] = _rate(group, "correct")
            metrics[f"correct_{level}"] = _count(group, "correct")
            metrics[f"total_{level}"] = len(group)
    if unanswerable:
        metrics["correct_refusal_rate"] = _rate(unanswerable, "correct_refusal")
        metrics["correct_refusals"] = _count(unanswerable, "correct_refusal")
        metrics["unanswerable_total"] = len(unanswerable)
    for outcome in (
        "wrong_value",
        "false_refusal",
        "missed_refusal",
        "no_submission",
        "api_error",
    ):
        metrics[outcome] = _count(outcomes, outcome)
    blocked = [rule for o in outcomes for rule in o.blocked_rules]
    metrics["blocked_queries"] = len(blocked)
    for rule in sorted(set(blocked)):
        metrics[f"blocked_{rule}"] = blocked.count(rule)
    calls = [c for o in outcomes for c in o.sql_calls]
    metrics["sql_calls"] = len(calls)
    metrics["sql_errors"] = sum(1 for c in calls if not c["ok"] and c["rule"] is None)
    measured = [o for o in outcomes if o.outcome != "api_error"]
    if measured:
        latencies = [o.latency_s for o in measured]
        metrics["latency_mean_s"] = round(statistics.mean(latencies), 3)
        metrics["latency_median_s"] = round(statistics.median(latencies), 3)
        metrics["latency_max_s"] = round(max(latencies), 3)
        metrics["steps_mean"] = round(statistics.mean(o.steps for o in measured), 3)
    metrics["cost_usd"] = round(sum(o.cost_usd for o in outcomes), 6)
    metrics["input_tokens"] = sum(o.input_tokens for o in outcomes)
    metrics["output_tokens"] = sum(o.output_tokens for o in outcomes)
    return metrics


def _count(outcomes: list[SQLCaseOutcome], outcome: str) -> int:
    return sum(1 for o in outcomes if o.outcome == outcome)


def _rate(outcomes: list[SQLCaseOutcome], outcome: str) -> float:
    return round(_count(outcomes, outcome) / len(outcomes), 4)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Évaluation de l'agent text-to-SQL.")
    parser.add_argument(
        "--scripted", action="store_true", help="client simulé, sans appel payant"
    )
    parser.add_argument("--ids", nargs="*", help="sous-ensemble de questions (ids)")
    parser.add_argument(
        "--model", help="modèle de l'agent (défaut : modèle de synthèse)"
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    configure_logging()

    cases = load_sql_eval()
    if args.ids:
        cases = [c for c in cases if c.id in set(args.ids)]
    ledger = CostLedger()
    executor = SafeSQLExecutor(
        max_rows=settings.data_agent_max_rows, timeout_s=settings.data_agent_timeout_s
    )
    client: Any
    if args.scripted:
        client, model = ScriptedSQLClient(cases), SCRIPTED_MODEL
    else:
        if not settings.anthropic_api_key:
            raise SystemExit("ANTHROPIC_API_KEY absente (ou utiliser --scripted).")
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        model = args.model or settings.data_agent_model or model_for(LLMRole.SYNTHESIS)
    agent = DataAgent(client, model, executor, max_steps=args.max_steps, ledger=ledger)

    outcomes = run_sql_eval(agent, cases)
    metrics = summarize(outcomes)
    mode = "scripted" if args.scripted else "real"
    report = {
        "mode": mode,
        "model": model,
        "questions": len(cases),
        "metrics": metrics,
        "details": [asdict(o) for o in outcomes],
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output:
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if not args.no_mlflow:
        log_data_agent_run(
            params={
                "mode": mode,
                "model": model,
                "questions": len(cases),
                "max_steps": args.max_steps,
                "max_rows": executor.max_rows,
                "timeout_s": executor.timeout_s,
            },
            metrics=metrics,
            artifacts=[args.output] if args.output else [],
        )


if __name__ == "__main__":
    main()
