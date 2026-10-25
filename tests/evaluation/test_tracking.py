"""Tests du tracking MLflow des runs d'évaluation."""

import importlib.util
from pathlib import Path

import pytest

from document_intelligence.evaluation.models import EvaluationReport, ExampleResult
from document_intelligence.evaluation.tracking import log_evaluation_run

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlflow") is None, reason="mlflow non installé"
)


def _report() -> EvaluationReport:
    return EvaluationReport(
        k=5,
        example_count=1,
        aggregate={"hit_rate": 1.0, "mrr": 1.0},
        results=[ExampleResult(question="q ?", answerable=True)],
    )


def test_logs_run_with_params_metrics_and_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")

    log_evaluation_run(
        _report(),
        retrieval_mode="hybrid",
        llm_model="claude-sonnet-5",
        rag_backend="langchain",
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{tmp_path / 'mlruns'}")
    experiment = client.get_experiment_by_name("rag-evaluation")
    assert experiment is not None

    (run,) = client.search_runs([experiment.experiment_id])
    assert run.data.params == {
        "retrieval_mode": "hybrid",
        "llm_model": "claude-sonnet-5",
        "k": "5",
        "dataset_size": "1",
        "rag_backend": "langchain",
    }
    assert run.data.metrics == {"hit_rate": 1.0, "mrr": 1.0}

    artifacts = client.list_artifacts(run.info.run_id)
    assert [a.path for a in artifacts] == ["evaluation_report.json"]


def test_logs_provided_report_path_as_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")

    log_evaluation_run(
        _report(),
        retrieval_mode="dense",
        llm_model=None,
        report_path=report_path,
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{tmp_path / 'mlruns'}")
    experiment = client.get_experiment_by_name("rag-evaluation")
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.data.params["llm_model"] == "none"

    artifacts = client.list_artifacts(run.info.run_id)
    assert [a.path for a in artifacts] == ["report.json"]


def test_logs_synthesis_model_and_cost_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    report = _report()
    report.cost = {
        "cost_actual_usd": 0.5,
        "cost_unoptimized_usd": 1.0,
        "cost_savings_pct": 50.0,
    }

    log_evaluation_run(
        report,
        retrieval_mode="hybrid",
        llm_model="claude-sonnet-5",
        synthesis_model="claude-haiku-4-5",
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{tmp_path / 'mlruns'}")
    experiment = client.get_experiment_by_name("rag-evaluation")
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.data.params["synthesis_model"] == "claude-haiku-4-5"
    assert run.data.metrics["cost_savings_pct"] == 50.0
    assert run.data.metrics["cost_unoptimized_usd"] == 1.0
    assert run.data.metrics["hit_rate"] == 1.0


def test_never_raises_when_mlflow_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        "document_intelligence.evaluation.tracking.importlib.util.find_spec",
        lambda name: None,
    )

    log_evaluation_run(_report(), retrieval_mode="hybrid", llm_model=None)

    assert "mlflow" in caplog.text
