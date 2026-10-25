"""Construction de l'exécuteur SQL et de l'agent de données depuis la config."""

from pathlib import Path

import pytest

from document_intelligence import services
from document_intelligence.core.config import settings
from document_intelligence.data_agent import DEFAULT_DB_PATH
from document_intelligence.orchestration.data_agent import DataAgent


def test_executor_uses_the_versioned_database_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "data_agent_db_path", None)
    monkeypatch.setattr(settings, "data_agent_max_rows", 50)
    executor = services.build_sql_executor()
    assert executor.db_path == DEFAULT_DB_PATH
    assert executor.max_rows == 50


def test_executor_rejects_a_missing_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "data_agent_db_path", str(tmp_path / "absent.sqlite"))
    with pytest.raises(FileNotFoundError):
        services.build_sql_executor()


def test_no_agent_without_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    monkeypatch.setattr(settings, "data_agent_db_path", None)
    assert services.build_data_agent(services.build_sql_executor()) is None


def test_agent_with_key_uses_the_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key-not-real")
    monkeypatch.setattr(settings, "data_agent_model", "claude-haiku-4-5")
    monkeypatch.setattr(settings, "data_agent_db_path", None)
    agent = services.build_data_agent(services.build_sql_executor())
    assert isinstance(agent, DataAgent)
    assert agent.model == "claude-haiku-4-5"
