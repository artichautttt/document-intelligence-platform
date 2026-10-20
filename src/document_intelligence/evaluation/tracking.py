"""Suivi MLflow des runs d'évaluation (tracking local, sans serveur dédié).

Le tracking se fait dans `file:./mlruns` par défaut (aucun serveur externe,
aucun appel réseau). `mlflow` est une dépendance optionnelle (groupe
`mlflow`) : si elle est absente, ou si l'écriture échoue (dossier en lecture
seule, disque plein...), le run est simplement consigné et l'évaluation
continue — la CI ne doit jamais échouer à cause du tracking.
"""

import importlib.util
import logging
import os
import tempfile
from pathlib import Path

from document_intelligence.evaluation.models import EvaluationReport

logger = logging.getLogger(__name__)

_DEFAULT_TRACKING_URI = "file:./mlruns"
_EXPERIMENT_NAME = "rag-evaluation"
_RELIABILITY_EXPERIMENT_NAME = "rag-judge-reliability"


def log_evaluation_run(
    report: EvaluationReport,
    *,
    retrieval_mode: str,
    llm_model: str | None,
    synthesis_model: str | None = None,
    report_path: Path | None = None,
) -> None:
    """Enregistre `report` comme un run MLflow ; ne lève jamais d'exception.

    Args:
        report: rapport produit par `evaluate`.
        retrieval_mode: méthode de retrieval utilisée ("hybrid" ou "dense").
        llm_model: modèle du LLM-juge, ou `None` si seul le retrieval a été évalué.
        synthesis_model: modèle de la synthèse RAG (model tiering), si différent
            du juge ou à tracer explicitement.
        report_path: rapport JSON détaillé déjà écrit sur disque, à joindre comme
            artefact. Sans lui, un fichier temporaire est écrit puis supprimé.
    """
    if importlib.util.find_spec("mlflow") is None:
        logger.warning(
            "mlflow n'est pas installé (groupe optionnel `mlflow`) : run non tracé."
        )
        return

    try:
        _log_run(
            report,
            retrieval_mode=retrieval_mode,
            llm_model=llm_model,
            synthesis_model=synthesis_model,
            report_path=report_path,
        )
    except Exception:
        logger.warning(
            "échec du tracking MLflow ; l'évaluation continue.", exc_info=True
        )


def _log_run(
    report: EvaluationReport,
    *,
    retrieval_mode: str,
    llm_model: str | None,
    synthesis_model: str | None,
    report_path: Path | None,
) -> None:
    # mlflow >= 3 exige cet opt-in explicite pour le backend fichier local
    # ("file:./mlruns") — c'est justement le mode demandé ici, sans serveur dédié.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(_EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_param("retrieval_mode", retrieval_mode)
        mlflow.log_param("llm_model", llm_model or "none")
        if synthesis_model is not None:
            mlflow.log_param("synthesis_model", synthesis_model)
        mlflow.log_param("k", report.k)
        mlflow.log_param("dataset_size", report.example_count)
        mlflow.log_metrics(report.aggregate)
        if report.cost:
            # Coût réel (tiering + cache) vs sans optimisation, pour chiffrer le gain.
            mlflow.log_metrics(report.cost)

        if report_path is not None:
            mlflow.log_artifact(str(report_path))
        else:
            _log_report_as_temp_artifact(report)


def log_reliability_run(
    *,
    params: dict[str, str | int | float],
    metrics: dict[str, float],
    artifacts: list[Path],
) -> bool:
    """Enregistre une mesure de fiabilité du juge par rubrique ; ne lève jamais.

    Un run par modèle juge, dans l'expérience `rag-judge-reliability`. Les
    métriques non calculables ne figurent pas dans `metrics` (rien n'est
    substitué) : elles restent visibles comme `null` dans le rapport JSON.

    Returns:
        `True` si le run a été écrit, `False` sinon (mlflow absent ou échec).
    """
    if importlib.util.find_spec("mlflow") is None:
        logger.warning(
            "mlflow n'est pas installé (groupe optionnel `mlflow`) : run non tracé."
        )
        return False
    try:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
        )
        mlflow.set_experiment(_RELIABILITY_EXPERIMENT_NAME)
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact))
    except Exception:
        logger.warning("échec du tracking MLflow ; la mesure continue.", exc_info=True)
        return False
    return True


def _log_report_as_temp_artifact(report: EvaluationReport) -> None:
    import mlflow

    from document_intelligence.evaluation.runner import dump_report

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evaluation_report.json"
        dump_report(report, path)
        mlflow.log_artifact(str(path))
