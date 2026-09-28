"""Tests du cache exact-match des réponses LLM (TTL, LRU, persistance, client)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from document_intelligence.orchestration.anthropic_client import AnthropicLLMClient
from document_intelligence.orchestration.cache import (
    LLMResponseCache,
    cache_key,
    normalize_text,
)
from document_intelligence.orchestration.cost import CostLedger


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Verdict(BaseModel):
    score: int


def _tool_response(payload: dict[str, object], tokens: tuple[int, int]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "submit_answer"
    block.input = payload
    response = MagicMock()
    response.content = [block]
    response.usage.input_tokens, response.usage.output_tokens = tokens
    return response


def _text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.usage.input_tokens, response.usage.output_tokens = 10, 2
    return response


class TestNormalization:
    def test_case_whitespace_and_unicode_forms_share_a_key(self) -> None:
        assert normalize_text("  Quel  est\tle CA ?\n") == "quel est le ca ?"
        assert cache_key("m", "Quel est le CA ?") == cache_key("m", "quel  est le ca ?")
        # NFKC : espace insécable et ligature « ﬁ » ramenés à leur forme canonique.
        assert cache_key("m", "ﬁn du contrat") == cache_key("m", "fin du contrat")

    def test_different_content_or_model_gives_different_keys(self) -> None:
        assert cache_key("m", "Quel est le CA ?") != cache_key(
            "m", "Quel est l'EBITDA ?"
        )
        assert cache_key("haiku", "q") != cache_key("sonnet", "q")
        assert cache_key("m", None, "q") != cache_key("m", "", "q")


class TestLLMResponseCache:
    def test_returns_stored_value_until_ttl_expires(self) -> None:
        clock = FakeClock()
        cache = LLMResponseCache(ttl_seconds=60, clock=clock)
        cache.set("k", {"a": 1}, input_tokens=100, output_tokens=20)

        clock.now += 59
        entry = cache.get("k")
        assert entry is not None
        assert (entry.value, entry.input_tokens, entry.output_tokens) == (
            {"a": 1},
            100,
            20,
        )

        clock.now += 1
        assert cache.get("k") is None
        assert len(cache) == 0

    def test_evicts_least_recently_used_beyond_max_entries(self) -> None:
        cache = LLMResponseCache(ttl_seconds=60, max_entries=2, clock=FakeClock())
        cache.set("a", 1, 0, 0)
        cache.set("b", 2, 0, 0)
        cache.get("a")  # « a » devient le plus récemment utilisé
        cache.set("c", 3, 0, 0)

        assert cache.get("b") is None
        assert cache.get("a") is not None
        assert cache.get("c") is not None

    def test_file_cache_survives_restart_without_expired_entries(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "cache" / "llm.json"
        clock = FakeClock()
        first = LLMResponseCache(ttl_seconds=60, path=path, clock=clock)
        first.set("fresh", "réponse", 5, 1)
        clock.now += 50
        first.set("late", "autre", 5, 1)

        clock.now += 20  # « fresh » a expiré, « late » est encore valide
        second = LLMResponseCache(ttl_seconds=60, path=path, clock=clock)
        assert second.get("fresh") is None
        late = second.get("late")
        assert late is not None and late.value == "autre"

    def test_corrupt_cache_file_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "llm.json"
        path.write_text("{pas du json", encoding="utf-8")

        cache = LLMResponseCache(ttl_seconds=60, path=path)

        assert len(cache) == 0
        cache.set("k", "v", 0, 0)
        assert cache.get("k") is not None


class TestClientCaching:
    def test_identical_structured_call_is_served_from_cache(self) -> None:
        ledger = CostLedger()
        client = AnthropicLLMClient(
            api_key="fake-key",
            model="claude-sonnet-5",
            role="synthesis",
            cache=LLMResponseCache(ttl_seconds=60),
            ledger=ledger,
        )
        with patch.object(
            client._client.messages,
            "create",
            return_value=_tool_response({"score": 4}, tokens=(1000, 50)),
        ) as create:
            first = client.complete_structured("Quel est le CA ?", Verdict, system="s")
            second = client.complete_structured(
                "quel est  le CA ?", Verdict, system="s"
            )

        assert first == second == Verdict(score=4)
        create.assert_called_once()
        assert [r.cached for r in ledger.records] == [False, True]
        # Le hit reporte les tokens de l'appel d'origine, pour chiffrer le gain.
        assert ledger.records[1].input_tokens == 1000

    def test_different_system_prompt_is_not_a_cache_hit(self) -> None:
        client = AnthropicLLMClient(
            api_key="fake-key", cache=LLMResponseCache(ttl_seconds=60)
        )
        with patch.object(
            client._client.messages,
            "create",
            return_value=_tool_response({"score": 4}, tokens=(1, 1)),
        ) as create:
            client.complete_structured("q", Verdict, system="a")
            client.complete_structured("q", Verdict, system="b")

        assert create.call_count == 2

    def test_invalid_output_is_not_cached(self) -> None:
        client = AnthropicLLMClient(
            api_key="fake-key", cache=LLMResponseCache(ttl_seconds=60)
        )
        invalid = _tool_response({"score": "pas un entier"}, tokens=(1, 1))
        valid = _tool_response({"score": 2}, tokens=(1, 1))
        with patch.object(
            client._client.messages, "create", side_effect=[invalid, valid, valid]
        ) as create:
            assert client.complete_structured("q", Verdict) == Verdict(score=2)
            assert client.complete_structured("q", Verdict) == Verdict(score=2)

        # 1 sortie invalide + 1 nouvelle tentative valide, puis un hit.
        assert create.call_count == 2

    def test_complete_is_cached(self) -> None:
        client = AnthropicLLMClient(
            api_key="fake-key", cache=LLMResponseCache(ttl_seconds=60)
        )
        with patch.object(
            client._client.messages, "create", return_value=_text_response("ok")
        ) as create:
            assert client.complete("Bonjour") == "ok"
            assert client.complete("bonjour") == "ok"

        create.assert_called_once()

    def test_without_cache_every_call_hits_the_api(self) -> None:
        client = AnthropicLLMClient(api_key="fake-key")
        with patch.object(
            client._client.messages, "create", return_value=_text_response("ok")
        ) as create:
            client.complete("Bonjour")
            client.complete("Bonjour")

        assert create.call_count == 2
