"""Exécution d'une évaluation de bout en bout sur un jeu de test."""

import json
import time
from pathlib import Path

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
)
from document_intelligence.evaluation.retrieval import is_relevant, retrieval_scores
from document_intelligence.orchestration.graph import answer_query
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.vectorization.store import VectorStore


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
            add("latency_ms_mean", result.generation.latency_ms)

    return {
        name: round(mean, 4)
        for name, values in columns.items()
        if (mean := _mean(values)) is not None
    }


def evaluate(
    examples: list[EvalExample],
    store: VectorStore,
    k: int = 5,
    llm: LLMClient | None = None,
    judge: LLMClient | None = None,
) -> EvaluationReport:
    """Évalue le retrieval et, si `llm` est fourni, la génération.

    Args:
        examples: jeu de test.
        store: vector store déjà alimenté avec les documents évalués.
        k: nombre de chunks récupérés par question.
        llm: client utilisé par le pipeline. Sans lui, seul le retrieval est évalué.
        judge: LLM-juge pour faithfulness/relevance (par défaut `llm`).
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
        result = ExampleResult(question=example.question, retrieval=retrieval)

        if llm is not None:
            started = time.perf_counter()
            answer = answer_query(example.question, store=store, llm=llm, k=k)
            latency_ms = (time.perf_counter() - started) * 1000

            texts = {r.chunk_id: r.text for r in retrieved}
            cited_relevant = [
                is_relevant(texts[c.chunk_id], example.relevant_substrings)
                for c in answer.citations
                if c.chunk_id in texts
            ]
            abstained = not answer.sufficient_context
            judged = judge is not None and not abstained
            result.answer = answer.answer
            result.generation = GenerationScores(
                faithfulness=(
                    faithfulness(answer.answer, [r.text for r in retrieved], judge)
                    if judged and judge is not None
                    else None
                ),
                answer_relevance=(
                    answer_relevance(example.question, answer.answer, judge)
                    if judged and judge is not None
                    else None
                ),
                citation_precision=(
                    citation_precision(cited_relevant) if example.answerable else None
                ),
                abstention_correct=abstained == (not example.answerable),
                latency_ms=latency_ms,
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
