"""Tests du framework d'évaluation RAG (métriques pures + runner avec doublures)."""

import re
import uuid
from pathlib import Path

import pytest

from document_intelligence.chunking.models import Chunk, ChunkProvenance
from document_intelligence.chunking.service import chunk_document
from document_intelligence.core.text import normalize
from document_intelligence.evaluation import EvalExample, evaluate, load_dataset
from document_intelligence.evaluation.corpus import build_extended_corpus
from document_intelligence.evaluation.generation import (
    FaithfulnessVerdict,
    RelevanceVerdict,
    citation_precision,
)
from document_intelligence.evaluation.retrieval import retrieval_scores
from document_intelligence.ingestion.service import ingest
from document_intelligence.orchestration.llm import LLMClient, T
from document_intelligence.vectorization.chroma_store import ChromaStore
from document_intelligence.vectorization.hybrid_store import HybridStore
from document_intelligence.vectorization.models import QueryResult
from document_intelligence.vectorization.store import VectorStore

# Seuils du garde-fou : fixés après mesure, voir README (section Évaluation).
HIT_RATE_FLOOR = 0.78  # mesuré : 0.836
MRR_FLOOR = 0.64  # mesuré : 0.698


def _qr(text: str) -> QueryResult:
    return QueryResult(
        chunk_id=str(uuid.uuid4()),
        text=text,
        score=1.0,
        provenance=ChunkProvenance(
            document_id="d", source_path="/tmp/x", element_ids=["e"]
        ),
    )


class TestRetrievalScores:
    def test_perfect_top_rank(self) -> None:
        scores = retrieval_scores(
            [_qr("48.2 millions"), _qr("bruit")], ["48.2 millions"]
        )

        assert scores.hit == 1.0
        assert scores.reciprocal_rank == 1.0
        assert scores.recall == 1.0
        assert scores.precision == 0.5
        assert scores.ndcg == 1.0

    def test_relevant_chunk_at_rank_three(self) -> None:
        results = [_qr("a"), _qr("b"), _qr("cible")]
        scores = retrieval_scores(results, ["cible"])

        assert scores.reciprocal_rank == pytest.approx(1 / 3)
        assert scores.ndcg == pytest.approx(0.5)  # 1/log2(4) = 0.5

    def test_miss(self) -> None:
        scores = retrieval_scores([_qr("a"), _qr("b")], ["cible"])

        assert (scores.hit, scores.recall, scores.reciprocal_rank, scores.ndcg) == (
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def test_recall_counts_covered_expected_passages(self) -> None:
        scores = retrieval_scores([_qr("passage un")], ["passage un", "passage deux"])

        assert scores.recall == 0.5

    def test_matching_ignores_accents_case_and_whitespace(self) -> None:
        assert normalize("  Délai   de\nPAIEMENT ") == "delai de paiement"
        scores = retrieval_scores([_qr("delai  de paiement")], ["Délai de paiement"])
        assert scores.hit == 1.0


class ScriptedLLM(LLMClient):
    """LLM + juge déterministe.

    Juge : 2 affirmations sur 3 soutenues, pertinence 5/5. Génération : cite le
    premier extrait, sauf si la question contient « penalites » (abstention).
    """

    def complete(self, prompt: str) -> str:
        raise AssertionError("complete_structured doit être utilisé")

    def complete_structured(
        self, prompt: str, schema: type[T], system: str | None = None
    ) -> T:
        if schema is FaithfulnessVerdict:
            payload: dict[str, object] = {
                "claims": [
                    {"claim": "a", "supported": True},
                    {"claim": "b", "supported": True},
                    {"claim": "c", "supported": False},
                ]
            }
        elif schema is RelevanceVerdict:
            payload = {"score": 5, "reasoning": "ok"}
        else:
            ids = re.findall(r'<extrait id="([^"]+)"', prompt)
            abstain = "penalites" in prompt.split("<question>")[-1]
            payload = {
                "answer": "réponse",
                "cited_chunk_ids": [] if abstain else ids[:1],
                "sufficient_context": not abstain,
            }
        return schema.model_validate(payload)


class ListStore(VectorStore):
    def __init__(self, texts: list[str]) -> None:
        self._chunks = [_qr(t) for t in texts]

    def add_chunks(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def query(self, text: str, k: int = 5) -> list[QueryResult]:
        return self._chunks[:k]


class TestEvaluate:
    def test_retrieval_only_without_llm(self) -> None:
        store = ListStore(["le chiffre 48.2 millions", "bruit"])
        examples = [EvalExample(question="ca ?", relevant_substrings=["48.2 millions"])]

        report = evaluate(examples, store, k=2)

        assert report.aggregate["hit_rate"] == 1.0
        assert report.aggregate["mrr"] == 1.0
        assert "faithfulness" not in report.aggregate

    def test_full_evaluation_with_judge_and_abstention(self) -> None:
        store = ListStore(["le chiffre 48.2 millions", "bruit"])
        examples = [
            EvalExample(question="ca ?", relevant_substrings=["48.2 millions"]),
            EvalExample(question="penalites ?", answerable=False),
        ]

        report = evaluate(examples, store, k=2, llm=ScriptedLLM())

        assert report.aggregate["faithfulness"] == pytest.approx(2 / 3, abs=1e-3)
        assert report.aggregate["answer_relevance"] == 1.0
        assert report.aggregate["citation_precision"] == 1.0
        assert report.aggregate["abstention_accuracy"] == 1.0
        assert report.aggregate["unanswerable_abstention_rate"] == 1.0
        assert report.aggregate["false_abstention_rate"] == 0.0
        # la question sans réponse n'entre pas dans les métriques de retrieval
        assert report.results[1].retrieval is None
        assert report.results[1].generation is not None
        assert report.results[1].generation.faithfulness is None

    def test_wrong_abstention_is_counted(self) -> None:
        store = ListStore(["contenu"])
        # question répondable mais le LLM s'abstient (mot-clé « penalites »)
        examples = [
            EvalExample(question="penalites ?", relevant_substrings=["contenu"])
        ]

        report = evaluate(examples, store, k=1, llm=ScriptedLLM())

        assert report.aggregate["abstention_accuracy"] == 0.0
        # le passage était dans le contexte : le refus est imputable à la génération
        assert report.aggregate["false_abstention_rate"] == 1.0


def test_citation_precision() -> None:
    assert citation_precision([True, False]) == 0.5
    assert citation_precision([]) is None


def test_load_dataset_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "ds.jsonl"
    path.write_text(
        '{"question": "q1", "relevant_substrings": ["a"]}\n\n'
        '{"question": "q2", "answerable": false}\n',
        encoding="utf-8",
    )

    examples = load_dataset(path)

    assert [e.question for e in examples] == ["q1", "q2"]
    assert examples[1].answerable is False


_GOLDEN = (
    Path(__file__).parents[2]
    / "src"
    / "document_intelligence"
    / "evaluation"
    / "golden"
)


_FIXTURES = Path(__file__).parents[1] / "fixtures" / "documents"


@pytest.fixture(scope="module")
def corpus_chunks(tmp_path_factory: pytest.TempPathFactory) -> list[Chunk]:
    """Chunks des fixtures et du corpus étendu (ingestion faite une seule fois)."""
    extended = build_extended_corpus(tmp_path_factory.mktemp("corpus"))
    paths = [*sorted(_FIXTURES.iterdir()), *extended]
    return [chunk for path in paths for chunk in chunk_document(ingest(path))]


def _golden_examples() -> list[EvalExample]:
    return [
        example
        for name in ("fixtures_qa.jsonl", "extended_qa.jsonl")
        for example in load_dataset(_GOLDEN / name)
    ]


def test_every_expected_passage_exists_in_the_corpus(
    corpus_chunks: list[Chunk],
) -> None:
    """Un jeu de test dont la réponse n'existe pas dans le corpus fausserait les métriques."""
    haystacks = [normalize(chunk.text) for chunk in corpus_chunks]
    missing = [
        (example.question, substring)
        for example in _golden_examples()
        for substring in example.relevant_substrings
        if not any(normalize(substring) in haystack for haystack in haystacks)
    ]

    assert missing == []


@pytest.fixture(scope="module")
def chroma_store(
    corpus_chunks: list[Chunk], tmp_path_factory: pytest.TempPathFactory
) -> ChromaStore:
    """Store dense partagé : l'embedding du corpus n'est calculé qu'une fois."""
    store = ChromaStore(persist_directory=str(tmp_path_factory.mktemp("chroma")))
    store.add_chunks(corpus_chunks)
    return store


def test_golden_set_retrieval_quality_gate(chroma_store: ChromaStore) -> None:
    """Garde-fou de régression sur le retrieval hybride (63 questions, k=5).

    Les seuils laissent une marge sous les valeurs mesurées à l'introduction
    (voir README) tout en détectant une vraie régression.
    """
    report = evaluate(_golden_examples(), HybridStore(chroma_store), k=5)

    assert report.aggregate["hit_rate"] >= HIT_RATE_FLOOR
    assert report.aggregate["mrr"] >= MRR_FLOOR


def test_hybrid_retrieval_is_not_worse_than_dense(chroma_store: ChromaStore) -> None:
    """Le réordonnancement lexical doit justifier sa présence sur le jeu de test."""
    examples = _golden_examples()

    dense = evaluate(examples, chroma_store, k=5)
    hybrid = evaluate(examples, HybridStore(chroma_store), k=5)

    assert hybrid.aggregate["hit_rate"] >= dense.aggregate["hit_rate"]
    assert hybrid.aggregate["mrr"] >= dense.aggregate["mrr"]
