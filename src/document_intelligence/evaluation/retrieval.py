"""Métriques de retrieval : hit rate, recall, precision, MRR, nDCG.

Fonctions pures, sans dépendance au vector store : elles évaluent une liste
ordonnée de résultats déjà retrouvés contre les passages attendus.
"""

import math

from document_intelligence.core.text import normalize
from document_intelligence.evaluation.models import RetrievalScores
from document_intelligence.vectorization.models import QueryResult


def is_relevant(text: str, relevant_substrings: list[str]) -> bool:
    """Vrai si `text` contient l'une des sous-chaînes attendues."""
    haystack = normalize(text)
    return any(normalize(s) in haystack for s in relevant_substrings)


def retrieval_scores(
    results: list[QueryResult], relevant_substrings: list[str]
) -> RetrievalScores:
    """Calcule les métriques de retrieval pour une question.

    - hit : au moins un chunk pertinent dans les résultats (hit rate@k).
    - recall : part des passages attendus couverts par au moins un chunk.
    - precision : part des chunks retrouvés qui sont pertinents.
    - reciprocal_rank : 1/rang du premier chunk pertinent (moyenné = MRR).
    - ndcg : nDCG@k à pertinence binaire ; l'idéal suppose autant de chunks
      pertinents que de passages attendus (plafonné à k).
    """
    flags = [is_relevant(r.text, relevant_substrings) for r in results]
    relevant_count = sum(flags)

    covered = sum(
        1
        for substring in relevant_substrings
        if any(is_relevant(r.text, [substring]) for r in results)
    )
    first_rank = next((i for i, flag in enumerate(flags, start=1) if flag), None)

    dcg = sum(1 / math.log2(rank + 1) for rank, flag in enumerate(flags, 1) if flag)
    ideal_count = min(len(relevant_substrings), len(results))
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

    return RetrievalScores(
        hit=1.0 if relevant_count else 0.0,
        recall=covered / len(relevant_substrings) if relevant_substrings else 0.0,
        precision=relevant_count / len(results) if results else 0.0,
        reciprocal_rank=1 / first_rank if first_rank else 0.0,
        ndcg=dcg / idcg if idcg else 0.0,
    )
