"""Modèles du framework d'évaluation RAG : jeu de test, scores, rapport."""

from pydantic import BaseModel, Field


class EvalExample(BaseModel):
    """Une question du jeu de test ("golden set") et ce qu'on en attend.

    La pertinence d'un chunk est définie par le texte (`relevant_substrings`)
    et non par son identifiant, car les identifiants sont régénérés à chaque
    ingestion. Un chunk est pertinent s'il contient l'une des sous-chaînes.
    """

    question: str
    relevant_substrings: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    answerable: bool = True


class RetrievalScores(BaseModel):
    """Métriques de retrieval pour une question."""

    hit: float
    recall: float
    precision: float
    reciprocal_rank: float
    ndcg: float


class GenerationScores(BaseModel):
    """Métriques de génération pour une question ; `None` = non applicable."""

    faithfulness: float | None = None
    answer_relevance: float | None = None
    citation_precision: float | None = None
    abstention_correct: bool
    latency_ms: float


class ExampleResult(BaseModel):
    """Résultat détaillé d'une question évaluée."""

    question: str
    retrieval: RetrievalScores | None = None
    generation: GenerationScores | None = None
    answerable: bool = True
    answer: str | None = None
    errors: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """Rapport complet : moyennes par métrique et détail par question."""

    k: int
    example_count: int
    aggregate: dict[str, float]
    results: list[ExampleResult]
