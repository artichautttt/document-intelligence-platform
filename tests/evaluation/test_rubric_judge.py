"""Rubrique et LLM-juge par rubrique (juge simulé, aucun appel réseau)."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import anthropic
import pytest
from anthropic.types import TextBlock, ToolUseBlock
from pydantic import BaseModel

from document_intelligence.evaluation.rubric_judge import (
    DEFAULT_RUBRIC_PATH,
    JUDGE_OUTPUT_ATTEMPTS,
    JudgedChunk,
    RawJudgeResponse,
    RubricError,
    RubricInput,
    RubricJudge,
    RubricJudgeAPIError,
    RubricJudgeClient,
    RubricJudgeError,
    build_judge_prompt,
    load_rubric,
    strict_tool_schema,
    verdict_schema,
)
from document_intelligence.orchestration.cost import CostLedger
from document_intelligence.orchestration.exceptions import LLMGenerationError
from document_intelligence.orchestration.instrumented import RawCompletion

CRITERIA = [
    "sources_verifiables",
    "exactitude",
    "absence_hallucination",
    "refus_justifie",
    "concision_formalisme",
]

ITEM = RubricInput(
    question="Quel est le chiffre d'affaires ?",
    answer="Le chiffre d'affaires est de 48,2 M€.",
    chunks=[
        JudgedChunk(chunk_id="c1", text="CA consolidé : 48.2 millions", cited=True),
        JudgedChunk(chunk_id="c2", text="Effectif : 120 personnes", cited=False),
    ],
)


def _verdict(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        cid: {"score": 2, "justification": "Conforme aux extraits."} for cid in CRITERIA
    }
    payload.update(overrides)
    return payload


def response(
    tool_input: Any | None,
    *,
    stop_reason: str = "tool_use",
    output_tokens: int = 200,
    text: str | None = None,
) -> RawJudgeResponse:
    """Réponse brute simulée ; `tool_input=None` = aucun appel d'outil."""
    blocks: list[dict[str, Any]] = []
    if text is not None:
        blocks.append({"type": "text", "text": text})
    if tool_input is not None:
        blocks.append(
            {"type": "tool_use", "name": "submit_answer", "input": tool_input}
        )
    return RawJudgeResponse(
        raw_output=json.dumps(blocks, ensure_ascii=False),
        tool_input=tool_input,
        stop_reason=stop_reason,
        input_tokens=1000,
        output_tokens=output_tokens,
    )


class ScriptedClient(RubricJudgeClient):
    """Vrai client du juge dont seul l'envoi HTTP est remplacé par un script.

    Chaque élément de `script` est une `RawJudgeResponse` à renvoyer, ou une
    exception à lever (erreur d'API simulée).
    """

    def __init__(
        self,
        script: list[RawJudgeResponse | Exception],
        model: str = "claude-haiku-4-5",
    ) -> None:
        super().__init__("fake-key", model, ledger=CostLedger())
        self.script = list(script)
        self.calls = 0

    def _send(
        self, prompt: str, schema: type[BaseModel], system: str | None
    ) -> RawCompletion:
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return RawCompletion(step, step.input_tokens, step.output_tokens)


def _judge(
    script: list[RawJudgeResponse | Exception],
) -> tuple[RubricJudge, ScriptedClient]:
    client = ScriptedClient(script)
    return RubricJudge(client, load_rubric()), client


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "rubric.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestRubric:
    def test_versioned_rubric_has_five_criteria_with_three_anchors(self) -> None:
        rubric = load_rubric(DEFAULT_RUBRIC_PATH)
        assert (rubric.id, rubric.version) == ("rag_answer", "v1")
        assert rubric.criterion_ids == CRITERIA
        for criterion in rubric.criteria:
            assert sorted(criterion.levels) == [0, 1, 2]
            assert all(anchor.strip() for anchor in criterion.levels.values())

    def test_missing_level_is_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "id: r\nversion: v1\ntitle: t\ncriteria:\n"
            "  - id: a\n    name: A\n    description: d\n    levels: {0: x, 2: z}\n",
        )
        with pytest.raises(RubricError, match="niveaux attendus"):
            load_rubric(path)

    def test_duplicate_criterion_is_rejected(self, tmp_path: Path) -> None:
        criterion = (
            "  - id: a\n    name: A\n    description: d\n"
            "    levels: {0: x, 1: y, 2: z}\n"
        )
        path = _write(
            tmp_path, f"id: r\nversion: v1\ntitle: t\ncriteria:\n{criterion}{criterion}"
        )
        with pytest.raises(RubricError, match="en double"):
            load_rubric(path)

    def test_empty_anchor_is_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "id: r\nversion: v1\ntitle: t\ncriteria:\n"
            "  - id: a\n    name: A\n    description: d\n"
            "    levels: {0: x, 1: ' ', 2: z}\n",
        )
        with pytest.raises(RubricError, match="ancre non vide"):
            load_rubric(path)

    def test_invalid_yaml_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(RubricError, match="illisible"):
            load_rubric(_write(tmp_path, "id: [non ferme\n"))

    def test_non_mapping_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(RubricError, match="objet YAML"):
            load_rubric(_write(tmp_path, "- a\n- b\n"))


class TestPromptAndSchema:
    def test_prompt_contains_rubric_question_chunks_and_answer(self) -> None:
        prompt = build_judge_prompt(ITEM, load_rubric())
        assert 'version="v1"' in prompt
        assert all(cid in prompt for cid in CRITERIA)
        assert '<extrait id="c1" source="" cite="oui">' in prompt
        assert 'cite="non"' in prompt
        assert "<reponse>Le chiffre d'affaires est de 48,2 M€.</reponse>" in prompt

    def test_schema_requires_every_criterion(self) -> None:
        schema = verdict_schema(load_rubric()).model_json_schema()
        assert schema["required"] == CRITERIA
        assert schema["additionalProperties"] is False

    def test_strict_api_schema_is_strict_compatible_and_same_structure(self) -> None:
        api_schema = strict_tool_schema(
            verdict_schema(load_rubric()).model_json_schema()
        )
        serialized = json.dumps(api_schema)
        # Mots-clés refusés par le mode strict, et aucune référence à résoudre.
        for keyword in (
            "minimum",
            "maximum",
            "minLength",
            "maxLength",
            "$ref",
            "$defs",
        ):
            assert f'"{keyword}"' not in serialized
        assert api_schema["required"] == CRITERIA
        assert api_schema["additionalProperties"] is False
        for criterion in CRITERIA:
            node = api_schema["properties"][criterion]
            assert node["type"] == "object"
            assert node["additionalProperties"] is False
            assert node["required"] == ["score", "justification"]
            assert node["properties"]["score"]["type"] == "integer"
            assert node["properties"]["score"]["enum"] == [0, 1, 2]
            assert node["properties"]["justification"]["type"] == "string"
        assert api_schema["properties"]["exactitude"]["description"] == "Exactitude"

    def test_strict_conversion_handles_refs_lists_and_scalars(self) -> None:
        schema = {
            "$defs": {"N": {"type": "integer", "minimum": 1, "maximum": 3}},
            "type": "object",
            "properties": {
                "n": {"$ref": "#/$defs/N", "description": "d"},
                "s": {"type": "string", "minLength": 2},
                "a": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]},
            },
        }
        assert strict_tool_schema(schema) == {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "d", "enum": [1, 2, 3]},
                "s": {"type": "string"},
                "a": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
        }


class TestRubricJudge:
    def test_valid_verdict_with_usage_and_attempt(self) -> None:
        judge, client = _judge(
            [response(_verdict(exactitude={"score": 1, "justification": "x"}))]
        )
        call = judge.judge(ITEM)
        assert call.scores == dict.fromkeys(CRITERIA, 2) | {"exactitude": 1}
        assert call.justifications["exactitude"] == "x"
        assert call.temperature == 0.0
        assert (call.usage.api_calls, call.usage.input_tokens) == (1, 1000)
        # claude-haiku-4-5 : 1000 × 1 $ + 200 × 5 $ par million de tokens.
        assert call.usage.cost_usd == pytest.approx(0.002)
        assert call.usage.latency_ms >= 0
        assert [a.validation_error for a in call.attempts] == [None]
        assert call.attempts[0].stop_reason == "tool_use"
        assert client.calls == 1

    @pytest.mark.parametrize(
        ("bad_input", "expected_error"),
        [
            (
                _verdict(exactitude={"score": 3, "justification": "x"}),
                "less than or equal",
            ),
            (
                _verdict(exactitude={"score": "2", "justification": "x"}),
                "valid integer",
            ),
            (
                _verdict(exactitude={"score": True, "justification": "x"}),
                "valid integer",
            ),
            (_verdict(exactitude={"score": 2, "justification": ""}), "at least 1"),
            (
                {k: v for k, v in _verdict().items() if k != "refus_justifie"},
                "refus_justifie",
            ),
            (_verdict(critere_inconnu={"score": 2, "justification": "x"}), "Extra"),
            ("pas du JSON {", "pas du JSON valide"),
        ],
        ids=[
            "hors-echelle",
            "chaine",
            "booleen",
            "justification-vide",
            "critere-manquant",
            "critere-inconnu",
            "json-invalide",
        ],
    )
    def test_invalid_output_raises_after_retry_never_default_score(
        self, bad_input: Any, expected_error: str
    ) -> None:
        judge, client = _judge([response(bad_input), response(bad_input)])
        with pytest.raises(RubricJudgeError, match="Verdict invalide") as error:
            judge.judge(ITEM)
        assert client.calls == JUDGE_OUTPUT_ATTEMPTS
        assert error.value.usage.api_calls == 2
        assert error.value.usage.cost_usd == pytest.approx(0.004)
        assert len(error.value.attempts) == 2
        for attempt in error.value.attempts:
            assert expected_error in (attempt.validation_error or "")

    def test_failed_attempt_keeps_raw_output_stop_reason_and_tokens(self) -> None:
        truncated = response(
            {"sources_verifiables": {"score": 2}},
            stop_reason="max_tokens",
            output_tokens=37,
            text="Je commence l'évaluation.",
        )
        judge, _ = _judge([truncated, truncated])
        with pytest.raises(RubricJudgeError) as error:
            judge.judge(ITEM)
        first = error.value.attempts[0]
        assert first.stop_reason == "max_tokens"
        assert first.output_tokens == 37
        assert json.loads(first.raw_output) == [
            {"type": "text", "text": "Je commence l'évaluation."},
            {
                "type": "tool_use",
                "name": "submit_answer",
                "input": {"sources_verifiables": {"score": 2}},
            },
        ]
        assert "justification" in (first.validation_error or "")

    def test_response_without_tool_call_is_an_output_failure(self) -> None:
        judge, _ = _judge(
            [
                response(None, stop_reason="end_turn", text="Voici mon avis."),
                response(None, stop_reason="end_turn", text="Voici mon avis."),
            ]
        )
        with pytest.raises(RubricJudgeError) as error:
            judge.judge(ITEM)
        assert "aucun appel d'outil" in (error.value.attempts[0].validation_error or "")
        assert "Voici mon avis." in error.value.attempts[0].raw_output

    def test_invalid_then_valid_output_keeps_both_attempts(self) -> None:
        judge, client = _judge([response("pas du JSON {"), response(_verdict())])
        call = judge.judge(ITEM)
        assert call.usage.api_calls == 2
        assert call.attempts[0].validation_error is not None
        assert call.attempts[1].validation_error is None
        assert client.calls == 2

    def test_api_error_keeps_previous_attempts(self) -> None:
        api_error = LLMGenerationError("crédit épuisé")
        judge, _ = _judge([response("pas du JSON {"), api_error])
        with pytest.raises(RubricJudgeAPIError, match="crédit épuisé") as error:
            judge.judge(ITEM)
        assert len(error.value.attempts) == 1
        assert error.value.usage.api_calls == 1

    def test_no_cache_between_passes(self) -> None:
        judge, client = _judge([response(_verdict()), response(_verdict())])
        judge.judge(ITEM)
        judge.judge(ITEM)
        assert client.calls == 2


def _tool_response(
    blocks: list[Any], stop_reason: str = "tool_use", output_tokens: int = 5
) -> MagicMock:
    api_response = MagicMock()
    api_response.content = blocks
    api_response.stop_reason = stop_reason
    api_response.usage.input_tokens = 10
    api_response.usage.output_tokens = output_tokens
    return api_response


def _tool_use_block(payload: dict[str, Any]) -> ToolUseBlock:
    return ToolUseBlock.model_construct(
        id="toolu_1", type="tool_use", name="submit_answer", input=payload
    )


class TestRubricJudgeClient:
    def _create_kwargs(self, model: str) -> dict[str, Any]:
        client = RubricJudgeClient("fake-key", model)
        judge = RubricJudge(client, load_rubric())
        # autospec : un argument absent de la signature du SDK fait échouer le test.
        with patch.object(
            client._client.messages,
            "create",
            autospec=True,
            return_value=_tool_response([_tool_use_block(_verdict())]),
        ) as create:
            judge.judge(ITEM)
        return dict(create.call_args.kwargs)

    def test_temperature_zero_when_model_accepts_it(self) -> None:
        kwargs = self._create_kwargs("claude-haiku-4-5")
        assert kwargs["extra_body"] == {"temperature": 0.0}

    def test_temperature_omitted_when_model_rejects_it(self) -> None:
        kwargs = self._create_kwargs("claude-sonnet-5")
        assert kwargs["extra_body"] is None

    @pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-haiku-4-5"])
    def test_tool_is_strict_with_api_schema_and_still_forced(self, model: str) -> None:
        kwargs = self._create_kwargs(model)
        [tool] = kwargs["tools"]
        assert tool["strict"] is True
        assert tool["input_schema"] == strict_tool_schema(
            verdict_schema(load_rubric()).model_json_schema()
        )
        assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_answer"}

    def test_client_side_validation_is_unchanged_by_strict_mode(self) -> None:
        # L'API applique le schéma assoupli ; Pydantic garde ses contraintes.
        too_long = _verdict(exactitude={"score": 2, "justification": "x" * 1001})
        judge, _ = _judge([response(too_long), response(too_long)])
        with pytest.raises(RubricJudgeError) as error:
            judge.judge(ITEM)
        assert "at most 1000" in (error.value.attempts[0].validation_error or "")

    def test_raw_output_serializes_every_sdk_block(self) -> None:
        client = RubricJudgeClient("fake-key", "claude-sonnet-5")
        text = TextBlock.model_construct(type="text", text="Analyse en cours")
        partial = _tool_use_block({"exactitude": {"score": 2}})
        with patch.object(
            client._client.messages,
            "create",
            autospec=True,
            return_value=_tool_response(
                [text, partial], stop_reason="max_tokens", output_tokens=36
            ),
        ):
            raw = client.request("prompt", verdict_schema(load_rubric()), None)
        assert raw.stop_reason == "max_tokens"
        assert raw.output_tokens == 36
        assert raw.tool_input == {"exactitude": {"score": 2}}
        blocks = json.loads(raw.raw_output)
        assert blocks[0]["text"] == "Analyse en cours"
        assert blocks[1]["input"] == {"exactitude": {"score": 2}}

    def test_api_error_is_propagated_with_context(self) -> None:
        client = RubricJudgeClient("fake-key", "claude-haiku-4-5")
        judge = RubricJudge(client, load_rubric())
        api_error = anthropic.APIError(message="boom", request=MagicMock(), body=None)
        with (
            patch.object(client._client.messages, "create", side_effect=api_error),
            pytest.raises(RubricJudgeAPIError, match="tentative 1"),
        ):
            judge.judge(ITEM)
