"""Métriques Prometheus du pipeline RAG, exposées par `GET /metrics/prometheus`.

Même principe que l'instrumentation du Mini-SIEM (`prometheus_client`,
registre par défaut, format texte), appliqué au pipeline RAG :

- `rag_queries_total{outcome}` : requêtes `/query` par issue — `answered`,
  `refused` (le modèle déclare `sufficient_context=false`) ou `error` ;
- `rag_query_duration_seconds{outcome}` : histogramme de latence par requête ;
- `rag_query_cost_usd` : histogramme du coût LLM **par requête**, somme des
  appels LLM faits pendant la requête au tarif du modèle effectivement appelé
  (tiering compris ; un hit de cache coûte 0) ;
- `rag_llm_cost_usd_total`, `rag_llm_tokens_total`, `rag_llm_calls_total`
  `{provider,model,role}` : cumul par fournisseur, modèle et rôle ;
- `rag_answer_verification_total{result}` : verdicts de l'agent de vérification.

Le coût par requête est accumulé dans une `ContextVar` ouverte par
`observe_query` : seuls les appels LLM faits dans ce contexte y sont comptés.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

__all__ = [
    "CONTENT_TYPE_LATEST",
    "QueryObservation",
    "observe_query",
    "record_llm_cost",
    "record_verification",
    "render_latest",
]

RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Requêtes RAG traitées, par issue (answered, refused, error)",
    ["outcome"],
)
RAG_QUERY_DURATION_SECONDS = Histogram(
    "rag_query_duration_seconds",
    "Latence de bout en bout d'une requête RAG (secondes)",
    ["outcome"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 20, 30, 60),
)
RAG_QUERY_COST_USD = Histogram(
    "rag_query_cost_usd",
    "Coût LLM d'une requête RAG (USD, tarif du modèle appelé, cache = 0)",
    buckets=(0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
RAG_LLM_COST_USD_TOTAL = Counter(
    "rag_llm_cost_usd_total",
    "Coût LLM cumulé (USD), par fournisseur, modèle et rôle",
    ["provider", "model", "role"],
)
RAG_LLM_TOKENS_TOTAL = Counter(
    "rag_llm_tokens_total",
    "Tokens LLM consommés, par fournisseur, modèle, rôle et sens",
    ["provider", "model", "role", "direction"],
)
RAG_LLM_CALLS_TOTAL = Counter(
    "rag_llm_calls_total",
    "Appels LLM, par fournisseur, modèle, rôle et origine (api ou cache)",
    ["provider", "model", "role", "source"],
)
RAG_ANSWER_VERIFICATION_TOTAL = Counter(
    "rag_answer_verification_total",
    "Verdicts de l'agent de vérification (grounded, flagged)",
    ["result"],
)

_request_cost: ContextVar[list[float] | None] = ContextVar(
    "rag_request_cost", default=None
)


def record_llm_cost(
    *,
    provider: str,
    model: str,
    role: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    cached: bool,
) -> None:
    """Comptabilise un appel LLM ; `cost_usd` doit valoir 0 pour un hit de cache."""
    RAG_LLM_CALLS_TOTAL.labels(
        provider, model, role, "cache" if cached else "api"
    ).inc()
    if cached:
        return
    RAG_LLM_TOKENS_TOTAL.labels(provider, model, role, "input").inc(input_tokens)
    RAG_LLM_TOKENS_TOTAL.labels(provider, model, role, "output").inc(output_tokens)
    RAG_LLM_COST_USD_TOTAL.labels(provider, model, role).inc(cost_usd)
    accumulator = _request_cost.get()
    if accumulator is not None:
        accumulator[0] += cost_usd


def record_verification(grounded: bool) -> None:
    """Comptabilise un verdict de l'agent de vérification."""
    RAG_ANSWER_VERIFICATION_TOTAL.labels("grounded" if grounded else "flagged").inc()


@dataclass
class QueryObservation:
    """Issue d'une requête, renseignée par l'appelant dans `observe_query`."""

    refused: bool = False
    cost_usd: float = 0.0


@contextmanager
def observe_query() -> Iterator[QueryObservation]:
    """Mesure latence, issue et coût LLM d'une requête RAG.

    Une exception est comptée comme `error` puis relancée.
    """
    observation = QueryObservation()
    accumulator = [0.0]
    token = _request_cost.set(accumulator)
    outcome = "error"
    started = perf_counter()
    try:
        yield observation
        outcome = "refused" if observation.refused else "answered"
    finally:
        _request_cost.reset(token)
        observation.cost_usd = accumulator[0]
        RAG_QUERIES_TOTAL.labels(outcome).inc()
        RAG_QUERY_DURATION_SECONDS.labels(outcome).observe(perf_counter() - started)
        RAG_QUERY_COST_USD.observe(accumulator[0])


def render_latest() -> bytes:
    """Exposition texte de toutes les métriques du registre par défaut."""
    return generate_latest()
