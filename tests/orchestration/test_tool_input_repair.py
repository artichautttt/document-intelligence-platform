"""Réparation des sorties d'outil mal formées observées avec le vrai modèle, et
isolation des sorties invalides pendant une évaluation."""

import json

import pytest
from pydantic import BaseModel

from document_intelligence.evaluation import EvalExample, evaluate
from document_intelligence.evaluation.generation import (
    ClaimVerdict,
    FaithfulnessVerdict,
)
from document_intelligence.orchestration.exceptions import (
    LLMGenerationError,
    LLMOutputError,
)
from document_intelligence.orchestration.llm import (
    LLMClient,
    T,
    validate_tool_input,
)
from tests.evaluation.test_evaluation import ListStore

_CLAIMS = [
    {"claim": "Le CA est de 48,2 M", "supported": True},
    {"claim": "La marge est de 25 %", "supported": False},
]


class Note(BaseModel):
    title: str
    tags: list[str]


class TestValidateToolInput:
    def test_valid_payload_is_untouched(self) -> None:
        verdict = validate_tool_input(FaithfulnessVerdict, {"claims": _CLAIMS})

        assert verdict.claims[0] == ClaimVerdict(**_CLAIMS[0])

    def test_list_serialized_as_a_json_string_is_repaired(self) -> None:
        payload = {"claims": json.dumps(_CLAIMS)}

        assert len(validate_tool_input(FaithfulnessVerdict, payload).claims) == 2

    def test_whole_object_serialized_in_the_field_is_repaired(self) -> None:
        # Forme exacte observée : {"claims": '{"claims": [...]}'}
        payload = {"claims": json.dumps({"claims": _CLAIMS})}

        verdict = validate_tool_input(FaithfulnessVerdict, payload)

        assert [c.supported for c in verdict.claims] == [True, False]

    def test_text_fields_starting_with_a_bracket_are_never_altered(self) -> None:
        note = validate_tool_input(Note, {"title": "[brouillon]", "tags": ["a"]})

        assert note.title == "[brouillon]"

    def test_unrecoverable_payload_raises_output_error(self) -> None:
        with pytest.raises(LLMOutputError):
            validate_tool_input(FaithfulnessVerdict, {"claims": "pas du json"})

    def test_output_error_is_a_generation_error(self) -> None:
        assert issubclass(LLMOutputError, LLMGenerationError)


class FlakyJudge(LLMClient):
    """Génère normalement ; le juge de faithfulness renvoie une sortie invalide."""

    def __init__(self, fail_with: Exception) -> None:
        self._fail_with = fail_with

    def complete(self, prompt: str) -> str:
        raise AssertionError

    def complete_structured(
        self, prompt: str, schema: type[T], system: str | None = None
    ) -> T:
        if schema is FaithfulnessVerdict:
            raise self._fail_with
        if schema.__name__ == "RelevanceVerdict":
            return schema.model_validate({"score": 5, "reasoning": "ok"})
        import re

        ids = re.findall(r'<extrait id="([^"]+)"', prompt)
        return schema.model_validate(
            {"answer": "r", "cited_chunk_ids": ids[:1], "sufficient_context": True}
        )


class TestEvaluationResilience:
    def test_invalid_judge_output_does_not_abort_the_run(self) -> None:
        store = ListStore(["le chiffre 48.2 millions", "bruit"])
        examples = [
            EvalExample(question=f"q{i} ?", relevant_substrings=["48.2 millions"])
            for i in range(3)
        ]

        report = evaluate(
            examples, store, k=2, llm=FlakyJudge(LLMOutputError("json invalide"))
        )

        assert report.example_count == 3
        assert report.aggregate["output_errors"] == 3.0
        assert "faithfulness" not in report.aggregate  # aucun score valide
        assert (
            report.aggregate["answer_relevance"] == 1.0
        )  # les autres métriques restent
        assert all("faithfulness" in r.errors[0] for r in report.results)

    def test_api_errors_still_abort_the_run(self) -> None:
        store = ListStore(["le chiffre 48.2 millions"])
        examples = [EvalExample(question="q ?", relevant_substrings=["48.2 millions"])]

        with pytest.raises(LLMGenerationError, match="crédit"):
            evaluate(
                examples,
                store,
                k=1,
                llm=FlakyJudge(LLMGenerationError("crédit insuffisant")),
            )
