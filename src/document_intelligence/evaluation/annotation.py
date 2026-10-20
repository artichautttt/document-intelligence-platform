"""Échantillon d'annotation manuelle : tirage stratifié, export CSV, relecture des notes.

- **Échantillon** : toutes les questions sans réponse possible sont gardées
  (elles seules testent un refus justifié), le reste est tiré parmi les
  questions répondables, au prorata de chaque fichier du jeu de test, avec une
  graine fixe.
- **Fichier de référence** (`sample.json`) : question, réponse, chunks
  récupérés et citations, exactement ce que le juge recevra. Le juge lit ce
  fichier, jamais le texte du CSV (qu'un tableur peut altérer).
- **CSV d'annotation** : id, question, réponse, chunks, et une colonne vide par
  critère. Aucune note de juge n'y figure (annotation en aveugle). Séparateur
  `;` et UTF-8 avec BOM, pour une ouverture directe dans Excel en français.
"""

import csv
import io
import random
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from document_intelligence.evaluation.models import EvalExample
from document_intelligence.evaluation.rubric_judge import (
    SCORE_LEVELS,
    JudgedChunk,
    Rubric,
    RubricInput,
)

CSV_DELIMITER = ";"
BASE_COLUMNS = ("id", "question", "reponse", "chunks_sources")


class AnnotationError(ValueError):
    """CSV d'annotation incomplet ou invalide ; liste tous les problèmes relevés."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__(
            f"{len(problems)} problème(s) dans le CSV d'annotation :\n- "
            + "\n- ".join(problems)
        )
        self.problems = problems


class SourcedExample(BaseModel):
    """Question du jeu de test avec un identifiant stable (`fichier#ligne`)."""

    id: str
    source_file: str
    example: EvalExample


def load_golden_examples(paths: Sequence[Path]) -> list[SourcedExample]:
    """Charge les jeux JSONL ; l'id compte les lignes non vides à partir de 1."""
    examples: list[SourcedExample] = []
    for path in paths:
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for number, line in enumerate(lines, start=1):
            examples.append(
                SourcedExample(
                    id=f"{path.stem}#{number:02d}",
                    source_file=path.name,
                    example=EvalExample.model_validate_json(line),
                )
            )
    return examples


def stratum(example: SourcedExample) -> str:
    """Strate d'une question : fichier source × répondable ou non."""
    kind = "repondable" if example.example.answerable else "sans_reponse"
    return f"{example.source_file}:{kind}"


def _allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Répartit `total` au prorata de `sizes` (plus forts restes, plafonné)."""
    population = sum(sizes.values())
    if total > population:
        raise ValueError(f"échantillon de {total} demandé pour {population} items")
    if population == 0:
        return dict.fromkeys(sizes, 0)
    shares = {key: total * size / population for key, size in sizes.items()}
    quota = {key: int(share) for key, share in shares.items()}
    by_remainder = sorted(sizes, key=lambda key: (-(shares[key] - quota[key]), key))
    for key in by_remainder[: total - sum(quota.values())]:
        quota[key] += 1
    return quota


def stratified_sample(
    examples: Sequence[SourcedExample], n: int, seed: int
) -> list[SourcedExample]:
    """Tire `n` questions, reproductible pour une même graine.

    Toutes les questions sans réponse sont retenues (au prorata de leurs
    strates si elles dépassent `n`) ; le reste est réparti entre les strates
    répondables au prorata de leur taille. L'ordre d'origine est conservé.

    Raises:
        ValueError: si `n` dépasse le nombre de questions.
    """
    if n > len(examples):
        raise ValueError(f"échantillon de {n} demandé pour {len(examples)} questions")
    groups: dict[str, list[SourcedExample]] = {}
    for example in examples:
        groups.setdefault(stratum(example), []).append(example)

    unanswerable = {
        key: len(group)
        for key, group in groups.items()
        if not group[0].example.answerable
    }
    answerable = {
        key: len(group) for key, group in groups.items() if key not in unanswerable
    }
    kept_unanswerable = min(n, sum(unanswerable.values()))
    quota = _allocate(unanswerable, kept_unanswerable) | _allocate(
        answerable, n - kept_unanswerable
    )

    rng = random.Random(seed)
    chosen = {
        example.id
        for key in sorted(groups)
        for example in rng.sample(groups[key], quota[key])
    }
    return [example for example in examples if example.id in chosen]


class AnnotationChunk(BaseModel):
    """Chunk récupéré pour une question ; `cited` = cité par la réponse."""

    chunk_id: str
    source: str
    text: str
    cited: bool


class AnnotationItem(BaseModel):
    """Une réponse du pipeline à annoter (et à juger)."""

    id: str
    source_file: str
    answerable: bool
    question: str
    answer: str
    sufficient_context: bool
    chunks: list[AnnotationChunk]

    def judge_input(self) -> RubricInput:
        return RubricInput(
            question=self.question,
            answer=self.answer,
            chunks=[
                JudgedChunk(
                    chunk_id=c.chunk_id, text=c.text, cited=c.cited, source=c.source
                )
                for c in self.chunks
            ],
        )


class AnnotationSample(BaseModel):
    """Échantillon complet et sa provenance ; source de vérité du juge."""

    rubric_id: str
    rubric_version: str
    seed: int
    population_size: int
    strata: dict[str, int]
    synthesis_model: str
    k: int
    retrieval_mode: str
    created_at: str
    items: list[AnnotationItem]


def write_sample(sample: AnnotationSample, path: Path) -> None:
    path.write_text(sample.model_dump_json(indent=2), encoding="utf-8")


def load_sample(path: Path) -> AnnotationSample:
    return AnnotationSample.model_validate_json(path.read_text(encoding="utf-8"))


def render_chunks(item: AnnotationItem) -> str:
    """Chunks tels que présentés à l'annotateur, cités marqués [CITÉ]."""
    blocks = []
    for rank, chunk in enumerate(item.chunks, start=1):
        mark = "[CITÉ] " if chunk.cited else ""
        blocks.append(f"[{rank}] {mark}{chunk.source}\n{chunk.text}")
    return "\n\n".join(blocks)


def annotation_columns(rubric: Rubric) -> list[str]:
    return [*BASE_COLUMNS, *rubric.criterion_ids]


def write_annotation_csv(sample: AnnotationSample, rubric: Rubric, path: Path) -> None:
    """CSV à remplir : une ligne par réponse, colonnes de critères vides."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=CSV_DELIMITER, quoting=csv.QUOTE_ALL)
        writer.writerow(annotation_columns(rubric))
        for item in sample.items:
            writer.writerow(
                [
                    item.id,
                    item.question,
                    item.answer,
                    render_chunks(item),
                    *([""] * len(rubric.criteria)),
                ]
            )


def _decode(raw: bytes) -> str:
    """UTF-8 (avec ou sans BOM), sinon Windows-1252 (réenregistrement Excel)."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def read_annotations(
    path: Path, rubric: Rubric, expected_ids: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Notes manuelles par id puis par critère.

    Accepte `;` ou `,` comme séparateur. Chaque cellule de critère doit valoir
    0, 1 ou 2 : une cellule vide ou invalide n'est jamais remplacée.

    Raises:
        AnnotationError: colonnes manquantes, ids inattendus ou en double,
            cellules vides ou hors échelle (tous les problèmes sont listés).
    """
    text = _decode(path.read_bytes())
    header = text.splitlines()[0] if text else ""
    delimiter = CSV_DELIMITER if CSV_DELIMITER in header else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    columns = [c.strip() for c in reader.fieldnames or []]
    missing = [c for c in ("id", *rubric.criterion_ids) if c not in columns]
    if missing:
        raise AnnotationError([f"colonnes manquantes : {', '.join(missing)}"])

    allowed = {str(level) for level in SCORE_LEVELS}
    problems: list[str] = []
    notes: dict[str, dict[str, int]] = {}
    for line, row in enumerate(reader, start=2):
        cells = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        item_id = cells.get("id", "")
        if not item_id:
            problems.append(f"ligne {line} : id vide")
            continue
        if item_id in notes:
            problems.append(f"ligne {line} : id {item_id} en double")
            continue
        scores: dict[str, int] = {}
        for criterion in rubric.criterion_ids:
            value = cells.get(criterion, "")
            if value not in allowed:
                shown = repr(value) if value else "vide"
                problems.append(
                    f"ligne {line} ({item_id}) : {criterion} = {shown}, attendu 0, 1 ou 2"
                )
            else:
                scores[criterion] = int(value)
        notes[item_id] = scores

    expected = set(expected_ids)
    problems += [f"id absent du CSV : {i}" for i in sorted(expected - set(notes))]
    problems += [
        f"id inconnu de l'échantillon : {i}" for i in sorted(set(notes) - expected)
    ]
    if problems:
        raise AnnotationError(problems)
    return notes
