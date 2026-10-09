"""Métriques Prometheus de `/query` : issue, latence, coût par requête, vérification."""

import re
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from document_intelligence.api.app import create_app
from document_intelligence.core import config
from document_intelligence.observability.prometheus import record_llm_cost
from document_intelligence.orchestration.llm import LLMClient, T
from document_intelligence.orchestration.models import VerificationVerdict
from document_intelligence.orchestration.verification import AnswerVerifier
from document_intelligence.vectorization.models import QueryResult


def _value(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


class PricedFakeLLM(LLMClient):
    """LLM factice qui déclare un appel facturé, comme le ferait un vrai client."""

    def __init__(self, sufficient: bool = True) -> None:
        self.sufficient = sufficient

    def complete(self, prompt: str) -> str:
        raise NotImplementedError

    def complete_structured(
        self, prompt: str, schema: type[T], system: str | None = None
    ) -> T:
        record_llm_cost(
            provider="fake",
            model="fake-model",
            role="synthesis",
            input_tokens=1000,
            output_tokens=100,
            cost_usd=0.004,
            cached=False,
        )
        ids = re.findall(r'<extrait id="([^"]+)"', prompt)
        return schema.model_validate(
            {
                "answer": "réponse" if self.sufficient else "rien dans les extraits",
                "cited_chunk_ids": ids[:1] if self.sufficient else [],
                "sufficient_context": self.sufficient,
            }
        )


class RejectingVerifier(AnswerVerifier):
    def verify(
        self, query: str, answer: str, sources: list[QueryResult]
    ) -> VerificationVerdict:
        return VerificationVerdict(grounded=False, unsupported_claims=["x"])


ClientFactory = Callable[..., TestClient]


@pytest.fixture
def make_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ClientFactory]:
    """Fabrique de clients de test ; leur lifespan est fermé en fin de test."""
    monkeypatch.setattr(
        config.settings, "chroma_persist_directory", str(tmp_path / ".chroma")
    )
    with ExitStack() as stack:

        def factory(
            llm: LLMClient, verifier: AnswerVerifier | None = None
        ) -> TestClient:
            app = create_app(llm=llm, verifier=verifier)
            return stack.enter_context(TestClient(app))

        yield factory


def _upload(client: TestClient, simple_docx: Path) -> None:
    with simple_docx.open("rb") as f:
        response = client.post(
            "/documents", files={"file": ("sample.docx", f, "application/octet-stream")}
        )
    assert response.status_code == 201


def test_answered_query_records_latency_and_cost(
    make_client: ClientFactory, simple_docx: Path
) -> None:
    client = make_client(PricedFakeLLM())
    _upload(client, simple_docx)
    answered = _value("rag_queries_total", outcome="answered")
    latency_count = _value("rag_query_duration_seconds_count", outcome="answered")
    cost_sum = _value("rag_query_cost_usd_sum")
    cost_count = _value("rag_query_cost_usd_count")

    response = client.post("/query", json={"query": "resultats", "k": 2})

    assert response.status_code == 200
    assert _value("rag_queries_total", outcome="answered") - answered == 1
    assert (
        _value("rag_query_duration_seconds_count", outcome="answered") - latency_count
        == 1
    )
    assert _value("rag_query_cost_usd_count") - cost_count == 1
    assert _value("rag_query_cost_usd_sum") - cost_sum == pytest.approx(0.004)


def test_refusal_and_error_outcomes(
    make_client: ClientFactory, simple_docx: Path
) -> None:
    client = make_client(PricedFakeLLM(sufficient=False))
    _upload(client, simple_docx)
    refused = _value("rag_queries_total", outcome="refused")
    errors = _value("rag_queries_total", outcome="error")

    assert client.post("/query", json={"query": "resultats"}).status_code == 200
    assert client.post("/query", json={"query": "   "}).status_code == 400

    assert _value("rag_queries_total", outcome="refused") - refused == 1
    assert _value("rag_queries_total", outcome="error") - errors == 1


def test_verification_is_exposed_in_response_and_metrics(
    make_client: ClientFactory, monkeypatch: pytest.MonkeyPatch, simple_docx: Path
) -> None:
    monkeypatch.setattr(config.settings, "verification_max_regenerations", 1)
    client = make_client(PricedFakeLLM(), RejectingVerifier())
    _upload(client, simple_docx)
    flagged = _value("rag_answer_verification_total", result="flagged")
    cost_sum = _value("rag_query_cost_usd_sum")

    body = client.post("/query", json={"query": "resultats"}).json()

    assert body["verified"] is False
    assert body["regenerations"] == 1
    assert _value("rag_answer_verification_total", result="flagged") - flagged == 2
    # Deux synthèses (initiale + régénération) dans la même requête.
    assert _value("rag_query_cost_usd_sum") - cost_sum == pytest.approx(0.008)


def test_prometheus_endpoint_serves_text_format(make_client: ClientFactory) -> None:
    client = make_client(PricedFakeLLM())

    response = client.get("/metrics/prometheus")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    for name in (
        "rag_queries_total",
        "rag_query_duration_seconds",
        "rag_query_cost_usd",
    ):
        assert f"# TYPE {name}" in response.text
    # L'instantané JSON historique reste servi sur /metrics.
    assert "spans" in client.get("/metrics").json()
