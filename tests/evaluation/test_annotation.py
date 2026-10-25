"""Échantillon d'annotation : tirage stratifié reproductible, CSV en aveugle, relecture."""

from collections import Counter
from pathlib import Path

import pytest

from document_intelligence.evaluation.annotation import (
    AnnotationChunk,
    AnnotationError,
    AnnotationItem,
    AnnotationSample,
    SourcedExample,
    load_golden_examples,
    read_annotations,
    stratified_sample,
    stratum,
    write_annotation_csv,
)
from document_intelligence.evaluation.models import EvalExample
from document_intelligence.evaluation.rubric_judge import load_rubric

GOLDEN = Path("src/document_intelligence/evaluation/golden")
DATASETS = [GOLDEN / "fixtures_qa.jsonl", GOLDEN / "extended_qa.jsonl"]
RUBRIC = load_rubric()
CRITERIA = RUBRIC.criterion_ids


def _golden() -> list[SourcedExample]:
    return load_golden_examples(DATASETS)


class TestStratifiedSample:
    def test_golden_set_has_63_questions_with_stable_ids(self) -> None:
        examples = _golden()
        assert len(examples) == 63
        assert len({e.id for e in examples}) == 63
        assert examples[0].id == "fixtures_qa#01"

    def test_same_seed_same_sample(self) -> None:
        first = [e.id for e in stratified_sample(_golden(), n=30, seed=0)]
        second = [e.id for e in stratified_sample(_golden(), n=30, seed=0)]
        assert first == second
        assert len(first) == 30

    def test_other_seed_other_sample(self) -> None:
        first = {e.id for e in stratified_sample(_golden(), n=30, seed=0)}
        other = {e.id for e in stratified_sample(_golden(), n=30, seed=1)}
        assert first != other

    def test_keeps_every_unanswerable_question_and_balances_the_rest(self) -> None:
        examples = _golden()
        sample = stratified_sample(examples, n=30, seed=0)
        unanswerable = {e.id for e in examples if not e.example.answerable}
        assert unanswerable <= {e.id for e in sample}
        # 22 répondables réparties au prorata de 13 (fixtures) et 42 (étendu).
        assert Counter(stratum(e) for e in sample) == {
            "fixtures_qa.jsonl:sans_reponse": 2,
            "extended_qa.jsonl:sans_reponse": 6,
            "fixtures_qa.jsonl:repondable": 5,
            "extended_qa.jsonl:repondable": 17,
        }

    def test_keeps_original_order(self) -> None:
        examples = _golden()
        order = {e.id: i for i, e in enumerate(examples)}
        positions = [order[e.id] for e in stratified_sample(examples, n=30, seed=3)]
        assert positions == sorted(positions)

    def test_more_unanswerable_than_requested(self) -> None:
        examples = [
            SourcedExample(
                id=f"q{i}",
                source_file="a.jsonl",
                example=EvalExample(question=f"q{i}", answerable=i < 2),
            )
            for i in range(6)
        ]
        sample = stratified_sample(examples, n=3, seed=0)
        assert all(not e.example.answerable for e in sample)

    def test_sample_larger_than_population_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            stratified_sample(_golden(), n=64, seed=0)


def _sample() -> AnnotationSample:
    items = [
        AnnotationItem(
            id=f"fixtures_qa#0{i}",
            source_file="fixtures_qa.jsonl",
            answerable=True,
            question=f"Question {i} ; avec séparateur ?",
            answer=f"Réponse {i}, sur\ndeux lignes.",
            sufficient_context=True,
            chunks=[
                AnnotationChunk(
                    chunk_id="c1", source="a.pdf", text="texte 1", cited=True
                ),
                AnnotationChunk(
                    chunk_id="c2", source="b.docx", text="texte 2", cited=False
                ),
            ],
        )
        for i in (1, 2)
    ]
    return AnnotationSample(
        rubric_id=RUBRIC.id,
        rubric_version=RUBRIC.version,
        seed=0,
        population_size=63,
        strata={"fixtures_qa.jsonl:repondable": 2},
        synthesis_model="claude-sonnet-5",
        k=5,
        retrieval_mode="hybrid",
        created_at="2026-10-19T00:00:00+00:00",
        items=items,
    )


def _fill(path: Path, rows: dict[str, list[str]]) -> None:
    """Remplit les colonnes de critères comme le ferait l'annotateur."""
    import csv

    with path.open(encoding="utf-8-sig", newline="") as handle:
        table = list(csv.reader(handle, delimiter=";"))
    for row in table[1:]:
        row[4:] = rows[row[0]]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle, delimiter=";").writerows(table)


class TestAnnotationCsv:
    def test_blind_csv_has_empty_criterion_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "annotations.csv"
        write_annotation_csv(_sample(), RUBRIC, path)
        raw = path.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8-sig")
        assert text.splitlines()[0] == ";".join(
            f'"{c}"' for c in ["id", "question", "reponse", "chunks_sources", *CRITERIA]
        )
        assert "[1] [CITÉ] a.pdf\ntexte 1" in text
        assert "[2] b.docx\ntexte 2" in text
        with pytest.raises(AnnotationError) as error:
            read_annotations(path, RUBRIC, ["fixtures_qa#01", "fixtures_qa#02"])
        assert len(error.value.problems) == 2 * len(CRITERIA)
        assert "vide" in error.value.problems[0]

    def test_filled_csv_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "annotations.csv"
        write_annotation_csv(_sample(), RUBRIC, path)
        _fill(
            path,
            {
                "fixtures_qa#01": ["2", "2", "1", "2", "0"],
                "fixtures_qa#02": [" 0 ", "1", "2", "2", "2"],
            },
        )
        notes = read_annotations(path, RUBRIC, ["fixtures_qa#01", "fixtures_qa#02"])
        assert notes["fixtures_qa#01"] == dict(
            zip(CRITERIA, [2, 2, 1, 2, 0], strict=True)
        )
        assert notes["fixtures_qa#02"]["sources_verifiables"] == 0

    def test_invalid_values_and_ids_are_all_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "annotations.csv"
        write_annotation_csv(_sample(), RUBRIC, path)
        _fill(
            path,
            {
                "fixtures_qa#01": ["2", "3", "1.5", "2", "oui"],
                "fixtures_qa#02": ["2", "2", "2", "2", "2"],
            },
        )
        with pytest.raises(AnnotationError) as error:
            read_annotations(path, RUBRIC, ["fixtures_qa#01", "fixtures_qa#03"])
        problems = "\n".join(error.value.problems)
        assert "exactitude = '3'" in problems
        assert "absence_hallucination = '1.5'" in problems
        assert "concision_formalisme = 'oui'" in problems
        assert "id absent du CSV : fixtures_qa#03" in problems
        assert "id inconnu de l'échantillon : fixtures_qa#02" in problems

    def test_comma_separated_cp1252_csv_is_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "annotations.csv"
        header = ",".join(["id", "question", *CRITERIA])
        path.write_bytes(f"{header}\nq1,Échéance ?,2,2,2,1,0\n".encode("cp1252"))
        assert read_annotations(path, RUBRIC, ["q1"])["q1"]["concision_formalisme"] == 0

    def test_missing_criterion_column(self, tmp_path: Path) -> None:
        path = tmp_path / "annotations.csv"
        path.write_text("id;question\nq1;x\n", encoding="utf-8")
        with pytest.raises(AnnotationError, match="colonnes manquantes"):
            read_annotations(path, RUBRIC, ["q1"])
