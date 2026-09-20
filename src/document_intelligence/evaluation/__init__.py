"""Évaluation du pipeline RAG : métriques de retrieval et de génération (LLM-juge)."""

from document_intelligence.evaluation.models import (
    EvalExample,
    EvaluationReport,
    ExampleResult,
)
from document_intelligence.evaluation.runner import evaluate, load_dataset

__all__ = [
    "EvalExample",
    "EvaluationReport",
    "ExampleResult",
    "evaluate",
    "load_dataset",
]
