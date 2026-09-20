"""Recherche hybride : candidats denses réordonnés par fusion avec un score lexical.

Les embeddings capturent le sens mais peuvent classer un passage voisin devant
celui qui répond littéralement à la question (mêmes mots-clés, autre section).
`HybridStore` demande `oversample × k` candidats au store dense, les note avec
BM25 (corpus = les candidats), puis fusionne les deux classements par
*Reciprocal Rank Fusion*. Il s'applique à n'importe quel `VectorStore` et ne
garde aucun état supplémentaire : rien à réindexer.
"""

import math
import re
from collections import Counter

from document_intelligence.chunking.models import Chunk
from document_intelligence.core.text import normalize
from document_intelligence.vectorization.models import QueryResult
from document_intelligence.vectorization.store import VectorStore

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    (  # noqa: SIM905 - liste lisible en prose
        "le la les l de des du d un une et en a au aux que qui quel quelle "
        "quels quelles est sont pour par sur dans ce cette ces se sa son ses "
        "qu il elle ils on ne pas plus ou y"
    ).split()
)
_STEM_LENGTH = (
    6  # racine grossière : « recommande », « recommandé », « recommandations »
)


def tokenize(text: str) -> list[str]:
    """Tokens normalisés, sans mots vides, tronqués à une racine grossière."""
    return [
        token[:_STEM_LENGTH]
        for token in _TOKEN.findall(normalize(text))
        if token not in _STOPWORDS and len(token) > 1
    ]


def bm25_scores(
    query: str, documents: list[str], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """Score BM25 de chaque document pour `query` (IDF calculé sur `documents`)."""
    tokenized = [tokenize(doc) for doc in documents]
    query_terms = set(tokenize(query))
    n_docs = len(tokenized)
    avg_len = sum(len(t) for t in tokenized) / n_docs if n_docs else 0.0
    doc_freq = Counter(term for tokens in tokenized for term in set(tokens))

    scores: list[float] = []
    for tokens in tokenized:
        counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            norm = k1 * (1 - b + b * len(tokens) / avg_len) if avg_len else k1
            score += idf * tf * (k1 + 1) / (tf + norm)
        scores.append(score)
    return scores


def _ranks(scores: list[float]) -> list[int]:
    """Rang (1 = meilleur) de chaque élément ; les ex æquo gardent l'ordre d'entrée."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


class HybridStore(VectorStore):
    """Enveloppe un `VectorStore` dense avec un réordonnancement lexical."""

    def __init__(
        self, inner: VectorStore, oversample: int = 4, rrf_k: int = 60
    ) -> None:
        self._inner = inner
        self._oversample = oversample
        self._rrf_k = rrf_k

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self._inner.add_chunks(chunks)

    def query(self, text: str, k: int = 5) -> list[QueryResult]:
        """Retourne les `k` meilleurs chunks après fusion des classements.

        `score` conserve la valeur du store dense : l'ordre des résultats peut
        donc ne pas être monotone en `score`.
        """
        candidates = self._inner.query(text, k=k * self._oversample)
        if len(candidates) <= 1:
            return candidates[:k]

        dense_ranks = list(range(1, len(candidates) + 1))
        lexical_ranks = _ranks(bm25_scores(text, [c.text for c in candidates]))
        fused = [
            1 / (self._rrf_k + dense) + 1 / (self._rrf_k + lexical)
            for dense, lexical in zip(dense_ranks, lexical_ranks, strict=True)
        ]
        order = sorted(range(len(candidates)), key=lambda i: (-fused[i], i))
        return [candidates[i] for i in order[:k]]
