"""Cohérence du jeu de questions procédurales avec les processus d'exemple."""

from pathlib import Path

from document_intelligence.chunking.service import chunk_document
from document_intelligence.evaluation.retrieval import is_relevant
from document_intelligence.evaluation.runner import load_dataset
from document_intelligence.ingestion.service import ingest

DATASET = Path("src/document_intelligence/evaluation/golden/bpmn_qa.jsonl")
EXAMPLES = Path("examples/bpmn")


def _chunks() -> list[str]:
    return [
        chunk.text
        for path in sorted(EXAMPLES.glob("*.bpmn"))
        for chunk in chunk_document(ingest(path))
    ]


def test_dataset_counts() -> None:
    examples = load_dataset(DATASET)
    assert len(examples) == 24
    assert sum(e.answerable for e in examples) == 20


def test_every_expected_passage_is_exactly_one_task_chunk() -> None:
    chunks = _chunks()
    for example in load_dataset(DATASET):
        for substring in example.relevant_substrings:
            matches = [text for text in chunks if is_relevant(text, [substring])]
            assert len(matches) == 1, (example.question, substring, len(matches))


def test_unanswerable_questions_have_no_expected_passage() -> None:
    for example in load_dataset(DATASET):
        if not example.answerable:
            assert example.relevant_substrings == []
