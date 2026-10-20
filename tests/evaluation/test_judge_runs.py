"""Passes persistées du juge : sauvegarde après chaque évaluation, reprise, taux d'échec.

Faux client (aucun appel réseau) : les réponses du modèle et les erreurs
d'API sont scriptées.
"""

from pathlib import Path

import pytest

from document_intelligence.evaluation.annotation import (
    AnnotationChunk,
    AnnotationItem,
    AnnotationSample,
)
from document_intelligence.evaluation.judge_runs import (
    RunConfig,
    RunStore,
    failure_summary,
    open_store,
    record_key,
    run_passes,
)
from document_intelligence.evaluation.rubric_judge import (
    RawJudgeResponse,
    RubricJudge,
    RubricJudgeAPIError,
    load_rubric,
)
from document_intelligence.orchestration.exceptions import LLMGenerationError
from tests.evaluation.test_rubric_judge import ScriptedClient, _verdict, response

MODEL = "claude-haiku-4-5"
INVALID = response("pas du JSON {", stop_reason="max_tokens", output_tokens=36)


def _sample(n: int = 3) -> AnnotationSample:
    return AnnotationSample(
        rubric_id="rag_answer",
        rubric_version="v1",
        seed=0,
        population_size=63,
        strata={},
        synthesis_model="claude-sonnet-5",
        k=5,
        retrieval_mode="hybrid",
        created_at="2026-10-19T00:00:00+00:00",
        items=[
            AnnotationItem(
                id=f"q{i}",
                source_file="fixtures_qa.jsonl",
                answerable=True,
                question=f"Question {i} ?",
                answer=f"Réponse {i}.",
                sufficient_context=True,
                chunks=[
                    AnnotationChunk(chunk_id="c1", source="a.pdf", text="t", cited=True)
                ],
            )
            for i in range(1, n + 1)
        ],
    )


def _config(passes: int = 1) -> RunConfig:
    return RunConfig(
        sample_path="sample.json",
        sample_created_at="2026-10-19T00:00:00+00:00",
        rubric_id="rag_answer",
        rubric_version="v1",
        judges=[MODEL],
        passes=passes,
    )


def _run(
    script: list[RawJudgeResponse | Exception],
    store: RunStore,
    path: Path,
    passes: int = 1,
) -> ScriptedClient:
    client = ScriptedClient(script, model=MODEL)
    run_passes(
        RubricJudge(client, load_rubric()),
        _sample(),
        passes,
        store,
        path,
        progress=lambda _: None,
    )
    return client


class TestOpenStore:
    def test_existing_results_are_never_overwritten_without_resume(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "raw.json"
        RunStore(config=_config()).save(path)
        with pytest.raises(FileExistsError, match="--resume"):
            open_store(path, _config(), resume=False)

    def test_resume_requires_existing_results(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            open_store(tmp_path / "raw.json", _config(), resume=True)

    def test_resume_requires_same_configuration(self, tmp_path: Path) -> None:
        path = tmp_path / "raw.json"
        RunStore(config=_config(passes=1)).save(path)
        with pytest.raises(ValueError, match="configuration différente"):
            open_store(path, _config(passes=3), resume=True)


class TestRunPasses:
    def test_api_error_keeps_every_finished_evaluation_on_disk(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "raw.json"
        store = open_store(path, _config(), resume=False)
        credit = LLMGenerationError("Your credit balance is too low")
        with pytest.raises(RubricJudgeAPIError, match="credit balance"):
            _run([response(_verdict()), INVALID, INVALID, credit], store, path)

        saved = RunStore.load(path)
        assert {k: r.status for k, r in saved.evaluations.items()} == {
            record_key(MODEL, 1, "q1"): "ok",
            record_key(MODEL, 1, "q2"): "echec",
            record_key(MODEL, 1, "q3"): "interrompu",
        }
        assert saved.evaluations[record_key(MODEL, 1, "q1")].scores == dict.fromkeys(
            load_rubric().criterion_ids, 2
        )

    def test_failure_keeps_raw_output_stop_reason_tokens_and_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "raw.json"
        store = open_store(path, _config(), resume=False)
        _run(
            [INVALID, INVALID, response(_verdict()), response(_verdict())], store, path
        )

        failed = RunStore.load(path).evaluations[record_key(MODEL, 1, "q1")]
        assert failed.status == "echec"
        assert failed.scores is None
        assert failed.judge == MODEL and failed.pass_number == 1
        assert len(failed.attempts) == 2
        attempt = failed.attempts[0]
        assert attempt.raw_output == INVALID.raw_output
        assert attempt.stop_reason == "max_tokens"
        assert attempt.output_tokens == 36
        assert "pas du JSON valide" in (attempt.validation_error or "")

    def test_resume_skips_successes_and_replays_the_rest(self, tmp_path: Path) -> None:
        path = tmp_path / "raw.json"
        store = open_store(path, _config(), resume=False)
        credit = LLMGenerationError("credit")
        with pytest.raises(RubricJudgeAPIError):
            _run([response(_verdict()), INVALID, INVALID, credit], store, path)

        resumed = open_store(path, _config(), resume=True)
        client = _run([response(_verdict()), response(_verdict())], resumed, path)

        assert client.calls == 2  # q1 déjà réussi n'est pas repayé
        final = RunStore.load(path)
        assert {r.status for r in final.evaluations.values()} == {"ok"}
        assert sorted((r.item_id, r.status) for r in final.superseded) == [
            ("q2", "echec"),
            ("q3", "interrompu"),
        ]

    def test_failure_rate_counts_failures_and_never_scores_them(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "raw.json"
        store = open_store(path, _config(), resume=False)
        _run(
            [response(_verdict()), INVALID, INVALID, response(_verdict())], store, path
        )

        summary = failure_summary(store, MODEL)
        assert (summary.evaluations, summary.ok, summary.failures) == (3, 2, 1)
        assert summary.failure_rate == pytest.approx(1 / 3)
        assert summary.interrupted == 0
        failed = store.evaluations[record_key(MODEL, 1, "q2")]
        assert failed.scores is None and failed.justifications is None

    def test_replayed_failures_stay_counted_after_resume(self, tmp_path: Path) -> None:
        path = tmp_path / "raw.json"
        store = open_store(path, _config(), resume=False)
        _run(
            [response(_verdict()), INVALID, INVALID, response(_verdict())], store, path
        )
        resumed = open_store(path, _config(), resume=True)
        _run([response(_verdict())], resumed, path)

        summary = failure_summary(resumed, MODEL)
        assert summary.failures == 0
        assert summary.replayed_failures == 1
