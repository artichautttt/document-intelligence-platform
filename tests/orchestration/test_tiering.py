"""Tests du model tiering et de l'estimation de coût (réel vs sans optimisation)."""

import pytest

from document_intelligence.core.config import Settings
from document_intelligence.orchestration.cost import (
    CostLedger,
    LLMCallRecord,
    cost_usd,
    price_for,
)
from document_intelligence.orchestration.tiering import (
    LLMRole,
    build_llm_client,
    build_response_cache,
    model_for,
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ANTHROPIC_MODEL",
        "LLM_SYNTHESIS_MODEL",
        "LLM_JUDGE_MODEL",
        "LLM_LIGHT_MODEL",
        "LLM_CACHE_ENABLED",
        "LLM_CACHE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class TestModelFor:
    def test_defaults_keep_the_reference_model_for_synthesis_and_judge(self) -> None:
        config = _settings(anthropic_model="claude-sonnet-5")

        assert model_for(LLMRole.SYNTHESIS, config) == "claude-sonnet-5"
        assert model_for(LLMRole.JUDGE, config) == "claude-sonnet-5"
        assert model_for(LLMRole.LIGHT, config) == "claude-haiku-4-5"

    def test_each_role_is_configurable(self) -> None:
        config = _settings(
            anthropic_model="claude-sonnet-5",
            llm_synthesis_model="claude-haiku-4-5",
            llm_judge_model="claude-opus-5",
            llm_light_model="claude-sonnet-4-6",
        )

        assert model_for(LLMRole.SYNTHESIS, config) == "claude-haiku-4-5"
        assert model_for(LLMRole.JUDGE, config) == "claude-opus-5"
        assert model_for(LLMRole.LIGHT, config) == "claude-sonnet-4-6"

    def test_roles_are_read_from_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_SYNTHESIS_MODEL", "claude-haiku-4-5")

        assert model_for(LLMRole.SYNTHESIS, _settings()) == "claude-haiku-4-5"
        assert model_for(LLMRole.JUDGE, _settings()) == "claude-sonnet-5"

    def test_built_client_uses_the_role_model(self) -> None:
        config = _settings(llm_synthesis_model="claude-haiku-4-5")

        client = build_llm_client(LLMRole.SYNTHESIS, api_key="fake-key", config=config)

        assert client.model == "claude-haiku-4-5"


class TestBuildResponseCache:
    def test_disabled_cache_returns_none(self) -> None:
        assert build_response_cache(_settings(llm_cache_enabled=False)) is None

    def test_enabled_cache_is_in_memory_by_default(self) -> None:
        cache = build_response_cache(_settings())
        assert cache is not None and len(cache) == 0


class TestCost:
    def test_known_model_prices(self) -> None:
        assert cost_usd("claude-sonnet-5", 1_000_000, 100_000) == pytest.approx(3.0)
        assert cost_usd("claude-haiku-4-5", 1_000_000, 100_000) == pytest.approx(1.5)

    def test_unknown_model_falls_back_to_configured_prices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from document_intelligence.core import config

        monkeypatch.setattr(config.settings, "llm_input_price_per_mtok", 7.0)
        assert price_for("modele-inconnu").input_per_mtok == 7.0

    def test_summary_separates_actual_and_unoptimized_cost(self) -> None:
        ledger = CostLedger()
        # Synthèse sur Haiku, juge sur Sonnet, et un hit de cache sur la synthèse.
        ledger.record(
            LLMCallRecord("synthesis", "claude-haiku-4-5", 1_000_000, 0, False)
        )
        ledger.record(
            LLMCallRecord("synthesis", "claude-haiku-4-5", 1_000_000, 0, True)
        )
        ledger.record(LLMCallRecord("judge", "claude-sonnet-5", 1_000_000, 0, False))

        summary = ledger.summary(reference_model="claude-sonnet-5")

        assert summary["llm_calls"] == 3
        assert summary["llm_cache_hits"] == 1
        assert summary["cost_actual_usd"] == pytest.approx(1.0 + 2.0)
        assert summary["cost_unoptimized_usd"] == pytest.approx(3 * 2.0)
        assert summary["cost_savings_pct"] == pytest.approx(50.0)
        assert summary["cost_actual_usd_synthesis"] == pytest.approx(1.0)
        assert summary["cost_savings_pct_synthesis"] == pytest.approx(75.0)
        assert summary["cost_savings_pct_judge"] == pytest.approx(0.0)

    def test_empty_ledger_has_zero_cost_and_no_savings_ratio(self) -> None:
        summary = CostLedger().summary(reference_model="claude-sonnet-5")

        assert summary["cost_actual_usd"] == 0
        assert "cost_savings_pct" not in summary
