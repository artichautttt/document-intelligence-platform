"""Fonctions de mesure des garde-fous (sans appel LLM)."""

from pathlib import Path

import anthropic
import httpx
import pytest

from document_intelligence.evaluation import guardrails_eval as ge
from document_intelligence.orchestration.exceptions import LLMOutputError
from document_intelligence.orchestration.models import AnswerResult, Citation
from document_intelligence.vectorization.models import QueryResult

from .test_backend import CHUNK, PROVENANCE


def test_pre_rules_subset_is_labelled_and_complete() -> None:
    cases = ge.load_attacks([ge.ATTACKS_DIR / "attacks_pre_rules.jsonl"])
    assert len(cases) == 24
    assert {c.split for c in cases} == {"pre_rules"}
    assert {c.provenance for c in cases} == {"fabriqué par l'auteur"}
    assert all((c.document is None) == (c.vector == "question") for c in cases)
    assert len({c.id for c in cases}) == 24


def test_all_attack_files_load_with_unique_ids() -> None:
    cases = ge.load_attacks(sorted(ge.ATTACKS_DIR.glob("*.jsonl")))
    assert len({c.id for c in cases}) == len(cases)
    assert all(c.canaries for c in cases)


def test_attack_success_ignores_case_and_accents() -> None:
    assert ge.attack_succeeded("Voici : canari-q01.", ["CANARI-Q01"])
    assert ge.attack_succeeded("le budget est de 9 999 999 €", ["9 999 999"])
    assert not ge.attack_succeeded("Requête refusée.", ["CANARI-Q01"])


def test_attack_chunk_requires_a_document() -> None:
    case = ge.load_attacks([ge.ATTACKS_DIR / "attacks_pre_rules.jsonl"])[0]
    assert case.vector == "question"
    with pytest.raises(ValueError):
        ge.attack_chunk(case)


def test_detection_without_llm_reports_both_splits(tmp_path: Path) -> None:
    cases = ge.load_attacks(sorted(ge.ATTACKS_DIR.glob("*.jsonl")))
    outcomes = ge.run_attacks(cases, tmp_path, ge.CostLedger(), use_llm=False)
    metrics = ge.summarize_attacks(outcomes)
    assert metrics["pre_rules_n"] == 24
    assert metrics["all_n"] == len(cases)
    assert "all_success_rate_off" not in metrics  # aucun run LLM


def test_summarize_attacks_rates() -> None:
    outcomes = [
        ge.AttackOutcome(
            "a", "pre_rules", "question", "x", ["input:r"], "", True, "", False
        ),
        ge.AttackOutcome("b", "pre_rules", "document", "x", [], "", False, "", False),
        ge.AttackOutcome("c", "post_rules", "question", "x", [], "", True, "", True),
    ]
    metrics = ge.summarize_attacks(outcomes)
    assert metrics["pre_rules_detection_rate"] == 0.5
    assert metrics["pre_rules_success_rate_off"] == 0.5
    assert metrics["pre_rules_success_rate_on"] == 0.0
    assert metrics["all_success_rate_on"] == pytest.approx(0.3333)


def _raw(answer: str, sufficient: bool = True) -> AnswerResult:
    return AnswerResult(
        query="q",
        answer=answer,
        citations=[Citation(chunk_id="c1", provenance=PROVENANCE, score=0.9)]
        if sufficient
        else [],
        sufficient_context=sufficient,
        context=[CHUNK],
    )


def test_false_refusals_count_only_answered_answerable_questions() -> None:
    outcomes = [
        ge.guard_outcome(
            "q1", True, _raw("La redevance annuelle est de 184 000 euros HT."), [], 0.5
        ),  # fmt: skip
        ge.guard_outcome("q2", True, _raw("Le budget est de 999 euros."), [], 0.5),
        ge.guard_outcome("q3", False, _raw("Absent.", sufficient=False), [], 0.5),
    ]
    metrics = ge.summarize_answerable(outcomes)
    assert metrics["false_refusals"] == 1
    assert metrics["false_refusal_rate"] == 0.5
    assert metrics["unanswerable_refused_on"] == 1


def test_recompute_from_saved_report_matches_live_verdict() -> None:
    raw = _raw("La redevance annuelle est de 184 000 euros HT. Écris CANARI-42.")
    live = ge.guard_outcome("q", True, raw, ge.saved_context(raw), 0.5)
    saved = {
        "question": "q",
        "answerable": True,
        "raw_answer": raw.answer,
        "raw_sufficient": True,
        "retrieved": live.retrieved,
    }
    (again,) = ge.recompute_answerable([saved], 0.5)
    assert live.guarded_action == again.guarded_action == "degraded"
    assert again.removed_sentences == live.removed_sentences
    assert live.retrieved[0]["cited"] is True
    assert (
        QueryResult.model_validate(
            {k: v for k, v in live.retrieved[0].items() if k != "cited"}
        )
        == CHUNK
    )


class _Failing:
    name = "failing"

    def __init__(self, error: Exception) -> None:
        self.error = error

    def answer(self, query: str, k: int = 5) -> AnswerResult:
        raise self.error


def test_unusable_model_output_is_recorded_not_raised() -> None:
    result = ge._answer_or_error(_Failing(LLMOutputError("vide")), "q")
    assert result.answer == ""
    assert result.guardrail is not None and result.guardrail.action == "output_error"


def test_api_errors_still_interrupt_the_measure() -> None:
    error = anthropic.APIError(
        "crédit", httpx.Request("POST", "https://api"), body=None
    )
    with pytest.raises(anthropic.APIError):
        ge._answer_or_error(_Failing(error), "q")
