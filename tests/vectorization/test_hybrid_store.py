"""Tests du réordonnancement hybride (dense + BM25 fusionnés par RRF)."""

import uuid

import pytest

from document_intelligence.chunking.models import Chunk, ChunkProvenance
from document_intelligence.vectorization.hybrid_store import (
    HybridStore,
    bm25_scores,
    tokenize,
)
from document_intelligence.vectorization.models import QueryResult
from document_intelligence.vectorization.store import VectorStore


def _qr(text: str) -> QueryResult:
    return QueryResult(
        chunk_id=str(uuid.uuid4()),
        text=text,
        score=0.5,
        provenance=ChunkProvenance(
            document_id="d", source_path="/tmp/x", element_ids=["e"]
        ),
    )


class FixedOrderStore(VectorStore):
    """Store « dense » factice retournant toujours ses résultats dans un ordre fixe."""

    def __init__(self, results: list[QueryResult]) -> None:
        self._results = results
        self.requested_k: int | None = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def query(self, text: str, k: int = 5) -> list[QueryResult]:
        self.requested_k = k
        return self._results[:k]


class TestTokenize:
    def test_drops_stopwords_accents_and_reduces_to_stem(self) -> None:
        assert tokenize("Que recommande l'audit ?") == ["recomm", "audit"]
        assert tokenize("recommandé") == tokenize("recommandations")


class TestBm25:
    def test_scores_documents_containing_query_terms_higher(self) -> None:
        scores = bm25_scores(
            "délais de paiement",
            ["les délais de paiement fournisseurs", "la marge opérationnelle"],
        )

        assert scores[0] > 0
        assert scores[1] == 0

    def test_empty_corpus(self) -> None:
        assert bm25_scores("x", []) == []


class TestHybridStore:
    def test_lexical_match_can_overtake_the_dense_first_choice(self) -> None:
        # Le dense place « constats » en tête, sans aucun mot de la question ;
        # « recommandations » est 2e en dense ET 1er en lexical : la fusion le remonte.
        constats = _qr("Constats : retards de règlement des fournisseurs.")
        reco = _qr("Recommandations : il est recommandé de suivre les délais.")
        other = _qr("Périmètre de l'audit sur cinquante transactions.")
        store = HybridStore(FixedOrderStore([constats, reco, other]))

        results = store.query("Que recommande l'audit sur les délais ?", k=1)

        assert [r.chunk_id for r in results] == [reco.chunk_id]

    def test_dense_first_choice_wins_when_lexical_only_weakly_disagrees(self) -> None:
        # RRF est volontairement conservateur : un rang 1 dense n'est pas renversé
        # par un simple avantage lexical d'un candidat classé plus bas en dense.
        first = _qr("Constats : délais de paiement fournisseurs dépassés.")
        other = _qr("Périmètre de la mission sur cinquante transactions.")
        reco = _qr("Recommandations : il est recommandé de suivre les délais.")
        store = HybridStore(FixedOrderStore([first, other, reco]))

        results = store.query("Que recommande l'audit sur les délais ?", k=1)

        assert [r.chunk_id for r in results] == [first.chunk_id]

    def test_keeps_dense_order_when_lexical_signal_is_absent(self) -> None:
        first, second = _qr("alpha bêta"), _qr("gamma delta")
        store = HybridStore(FixedOrderStore([first, second]))

        results = store.query("zzz inconnu", k=2)

        assert [r.chunk_id for r in results] == [first.chunk_id, second.chunk_id]

    def test_oversamples_candidates_from_the_inner_store(self) -> None:
        inner = FixedOrderStore([_qr("a b"), _qr("c d")])

        HybridStore(inner, oversample=4).query("a", k=3)

        assert inner.requested_k == 12

    def test_single_or_empty_results_are_returned_as_is(self) -> None:
        assert HybridStore(FixedOrderStore([])).query("q", k=3) == []
        only = _qr("seul")
        assert HybridStore(FixedOrderStore([only])).query("q", k=3) == [only]

    def test_add_chunks_delegates_to_the_inner_store(self) -> None:
        class Recording(FixedOrderStore):
            def __init__(self) -> None:
                super().__init__([])
                self.added: list[Chunk] = []

            def add_chunks(self, chunks: list[Chunk]) -> None:
                self.added.extend(chunks)

        inner = Recording()
        chunk = Chunk(
            chunk_id="c",
            text="t",
            provenance=ChunkProvenance(
                document_id="d", source_path="/x", element_ids=["e"]
            ),
        )

        HybridStore(inner).add_chunks([chunk])

        assert inner.added == [chunk]

    def test_never_returns_more_than_k(self) -> None:
        results = [_qr(f"mot{i} commun") for i in range(10)]

        assert len(HybridStore(FixedOrderStore(results)).query("commun", k=3)) == 3


@pytest.mark.parametrize("k", [1, 5])
def test_hybrid_results_are_a_subset_of_dense_candidates(k: int) -> None:
    candidates = [_qr(f"texte numéro {i}") for i in range(20)]

    results = HybridStore(FixedOrderStore(candidates)).query("numéro 7", k=k)

    ids = {c.chunk_id for c in candidates}
    assert all(r.chunk_id in ids for r in results)
