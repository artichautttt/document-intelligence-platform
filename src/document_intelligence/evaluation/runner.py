"""Exécution d'une évaluation de bout en bout sur un jeu de test."""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from document_intelligence.evaluation.generation import (
    answer_relevance,
    citation_precision,
    faithfulness,
)
from document_intelligence.evaluation.models import (
    EvalExample,
    EvaluationReport,
    ExampleResult,
    GenerationScores,
    VerificationOutcome,
)
from document_intelligence.evaluation.retrieval import is_relevant, retrieval_scores
from document_intelligence.evaluation.verification import corrupt_answer
from document_intelligence.orchestration.exceptions import LLMOutputError
from document_intelligence.orchestration.graph import answer_query
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.orchestration.models import AnswerResult
from document_intelligence.orchestration.verification import AnswerVerifier
from document_intelligence.vectorization.models import QueryResult
from document_intelligence.vectorization.store import VectorStore

_R = TypeVar("_R")


def _guarded(errors: list[str], label: str, call: Callable[[], _R]) -> _R | None:
    """Exécute `call` ; une sortie de modèle invalide est consignée, pas propagée.

    Les erreurs d'API (clé invalide, crédit épuisé, réseau) restent fatales : les
    ignorer ferait brûler du crédit sur des appels voués à l'échec.
    """
    try:
        return call()
    except LLMOutputError as exc:
        errors.append(f"{label}: {exc}")
        return None


def load_dataset(path: Path) -> list[EvalExample]:
    """Charge un jeu de test JSONL (un `EvalExample` par ligne non vide)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalExample.model_validate_json(line) for line in lines if line.strip()]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(results: list[ExampleResult]) -> dict[str, float]:
    columns: dict[str, list[float]] = {}

    def add(name: str, value: float | bool | None) -> None:
        if value is not None:
            columns.setdefault(name, []).append(float(value))

    for result in results:
        if result.retrieval:
            add("hit_rate", result.retrieval.hit)
            add("recall", result.retrieval.recall)
            add("precision", result.retrieval.precision)
            add("mrr", result.retrieval.reciprocal_rank)
            add("ndcg", result.retrieval.ndcg)
        if result.generation:
            add("faithfulness", result.generation.faithfulness)
            add("answer_relevance", result.generation.answer_relevance)
            add("citation_precision", result.generation.citation_precision)
            add("abstention_accuracy", result.generation.abstention_correct)
            if not result.answerable:
                # Part des questions sans réponse correctement refusées.
                add(
                    "unanswerable_abstention_rate", result.generation.abstention_correct
                )
            elif result.retrieval and result.retrieval.hit:
                # Refus injustifié alors que le passage attendu était dans le
                # contexte : isole la génération des échecs de retrieval.
                add(
                    "false_abstention_rate",
                    not result.generation.abstention_correct,
                )
            add("latency_ms_mean", result.generation.latency_ms)
        if result.verification:
            outcome = result.verification
            add("verification_flag_rate", outcome.flagged)
            add("verification_final_grounded_rate", outcome.verified_final)
            if outcome.calibration_flagged is not None:
                add("calibration_detection_rate", outcome.calibration_flagged)

    aggregate = {
        name: round(mean, 4)
        for name, values in columns.items()
        if (mean := _mean(values)) is not None
    }
    output_errors = sum(len(r.errors) for r in results)
    if output_errors:
        aggregate["output_errors"] = float(output_errors)

    # Comptes bruts de la vérification, plus parlants que des taux sur un sous-ensemble.
    outcomes = [r.verification for r in results if r.verification]
    if outcomes:
        calibrated = [o for o in outcomes if o.calibration_flagged is not None]
        aggregate.update(
            {
                "verification_checked": float(len(outcomes)),
                "verification_flagged": float(sum(o.flagged for o in outcomes)),
                "verification_regenerations": float(
                    sum(o.regenerations for o in outcomes)
                ),
                "verification_final_flagged": float(
                    sum(o.verified_final is False for o in outcomes)
                ),
                "calibration_cases": float(len(calibrated)),
                "calibration_detected": float(
                    sum(bool(o.calibration_flagged) for o in calibrated)
                ),
            }
        )
    return aggregate


def _score_generation(
    example: EvalExample,
    retrieved: list[QueryResult],
    store: VectorStore,
    k: int,
    llm: LLMClient,
    judge: LLMClient | None,
    errors: list[str],
    verifier: AnswerVerifier | None = None,
    max_regenerations: int = 1,
) -> tuple[AnswerResult | None, GenerationScores | None]:
    """Exécute le pipeline sur une question et note la réponse.

    Une sortie de modèle invalide est consignée dans `errors` ; l'exemple garde
    alors des scores partiels au lieu d'interrompre toute l'évaluation.
    """
    started = time.perf_counter()
    answer = _guarded(
        errors,
        "synthesis",
        lambda: answer_query(
            example.question,
            store=store,
            llm=llm,
            k=k,
            verifier=verifier,
            max_regenerations=max_regenerations,
        ),
    )
    latency_ms = (time.perf_counter() - started) * 1000
    if answer is None:
        return None, None

    texts = {r.chunk_id: r.text for r in retrieved}
    cited_relevant = [
        is_relevant(texts[c.chunk_id], example.relevant_substrings)
        for c in answer.citations
        if c.chunk_id in texts
    ]
    abstained = not answer.sufficient_context
    contexts = [r.text for r in retrieved]

    faithfulness_score: float | None = None
    relevance_score: float | None = None
    if judge is not None and not abstained:
        faithfulness_score = _guarded(
            errors, "faithfulness", lambda: faithfulness(answer.answer, contexts, judge)
        )
        relevance_score = _guarded(
            errors,
            "answer_relevance",
            lambda: answer_relevance(example.question, answer.answer, judge),
        )

    return answer, GenerationScores(
        faithfulness=faithfulness_score,
        answer_relevance=relevance_score,
        citation_precision=(
            citation_precision(cited_relevant) if example.answerable else None
        ),
        abstention_correct=abstained == (not example.answerable),
        latency_ms=latency_ms,
    )


def _verification_outcome(
    example: EvalExample,
    answer: AnswerResult,
    retrieved: list[QueryResult],
    verifier: AnswerVerifier,
    calibrate: bool,
    errors: list[str],
) -> VerificationOutcome | None:
    """Résumé de la vérification ; avec `calibrate`, soumet une copie faussée."""
    if not answer.verifications:
        return None
    outcome = VerificationOutcome(
        verdicts=len(answer.verifications),
        flagged=any(not v.grounded for v in answer.verifications),
        regenerations=answer.regenerations,
        verified_final=answer.verified,
        unsupported_claims=[
            claim for v in answer.verifications for claim in v.unsupported_claims
        ],
        rejected_answers=answer.rejected_answers,
    )
    # La copie faussée part de la réponse initiale, avant toute régénération.
    initial = answer.rejected_answers[0] if answer.rejected_answers else answer.answer
    corrupted = corrupt_answer(initial) if calibrate else None
    if corrupted is not None:
        verdict = _guarded(
            errors,
            "calibration",
            lambda: verifier.verify(example.question, corrupted, retrieved),
        )
        if verdict is not None:
            outcome.calibration_corrupted_answer = corrupted
            outcome.calibration_flagged = not verdict.grounded
    return outcome


def evaluate(
    examples: list[EvalExample],
    store: VectorStore,
    k: int = 5,
    llm: LLMClient | None = None,
    judge: LLMClient | None = None,
    verifier: AnswerVerifier | None = None,
    max_regenerations: int = 1,
    calibrate_verifier: bool = False,
) -> EvaluationReport:
    """Évalue le retrieval et, si `llm` est fourni, la génération.

    Args:
        examples: jeu de test.
        store: vector store déjà alimenté avec les documents évalués.
        k: nombre de chunks récupérés par question.
        llm: client utilisé par le pipeline. Sans lui, seul le retrieval est évalué.
        judge: LLM-juge pour faithfulness/relevance (par défaut `llm`).
        verifier: agent de vérification inséré dans le pipeline (optionnel).
        max_regenerations: régénérations au plus après un verdict négatif.
        calibrate_verifier: soumet aussi au vérificateur une copie faussée de
            chaque réponse contenant un nombre (cf. `evaluation.verification`).
    """
    judge = judge or llm
    results: list[ExampleResult] = []

    for example in examples:
        retrieved = store.query(example.question, k=k)
        retrieval = (
            retrieval_scores(retrieved, example.relevant_substrings)
            if example.answerable and example.relevant_substrings
            else None
        )
        result = ExampleResult(
            question=example.question,
            answerable=example.answerable,
            retrieval=retrieval,
        )

        if llm is not None:
            answer, result.generation = _score_generation(
                example,
                retrieved,
                store,
                k,
                llm,
                judge,
                result.errors,
                verifier=verifier,
                max_regenerations=max_regenerations,
            )
            if answer is not None:
                result.answer = answer.answer
                if verifier is not None:
                    result.verification = _verification_outcome(
                        example,
                        answer,
                        retrieved,
                        verifier,
                        calibrate_verifier,
                        result.errors,
                    )

        results.append(result)

    return EvaluationReport(
        k=k,
        example_count=len(examples),
        aggregate=_aggregate(results),
        results=results,
    )


def format_report(report: EvaluationReport) -> str:
    """Rendu texte lisible des métriques agrégées."""
    lines = [f"Évaluation RAG — {report.example_count} questions, k={report.k}"]
    lines += [f"  {name:<22} {value:.3f}" for name, value in report.aggregate.items()]
    return "\n".join(lines)


def dump_report(report: EvaluationReport, path: Path) -> None:
    """Écrit le rapport détaillé au format JSON."""
    path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
