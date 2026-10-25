"""Jeu d'évaluation text-to-SQL : chargement et comparaison des réponses."""

import math
import re
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

DEFAULT_SQL_EVAL_PATH = Path(__file__).parent / "golden" / "sql" / "taxi_sql_eval.jsonl"

Difficulty = Literal["simple", "intermediaire", "difficile", "sans_reponse"]
DIFFICULTIES: tuple[Difficulty, ...] = (
    "simple",
    "intermediaire",
    "difficile",
    "sans_reponse",
)
RELATIVE_TOLERANCE = 0.005


class SQLEvalCase(BaseModel):
    """Une question du jeu d'évaluation (voir golden/sql/README.md)."""

    id: str
    difficulty: Difficulty
    provenance: str
    question: str
    answerable: bool
    answer_type: Literal["integer", "number", "text"] | None
    decimals: int | None
    expected: int | float | str | None
    reference_sql: str | None
    tie_check_sql: str | None = None
    why_unanswerable: str | None = None


def load_sql_eval(path: Path = DEFAULT_SQL_EVAL_PATH) -> list[SQLEvalCase]:
    return [
        SQLEvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


_THOUSANDS_COMMA = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")


def parse_number(value: object) -> float | None:
    """Nombre contenu dans `value` (nombre ou texte « 27,12 », « 3 330 486 $ »)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value) if math.isfinite(value) else None
    if not isinstance(value, str):
        return None
    text = re.sub(r"[\s  $€%]", "", value)
    if _THOUSANDS_COMMA.match(text):
        text = text.replace(",", "")
    elif "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def normalize_text(value: object) -> str:
    """Texte comparable : sans accents, casse ni ponctuation finale."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.casefold().split()).strip(" .,;:!?'\"«»")


def value_matches(case: SQLEvalCase, value: object) -> bool:
    """La valeur rendue correspond-elle au résultat de la requête de référence ?

    - entier : égalité exacte ;
    - nombre : écart au plus max(0,5 % de la valeur attendue, une demi-unité de
      la précision demandée), pour accepter un arrondi correct ;
    - texte : égalité après normalisation (accents, casse, espaces).
    """
    if not case.answerable or case.expected is None or value is None:
        return False
    if case.answer_type == "text":
        return normalize_text(value) == normalize_text(case.expected)
    number = parse_number(value)
    expected = parse_number(case.expected)
    if number is None or expected is None:
        return False
    if case.answer_type == "integer":
        return abs(number - expected) < 1e-9
    decimals = case.decimals if case.decimals is not None else 2
    half_unit = 0.5 * 10.0**-decimals
    tolerance = max(RELATIVE_TOLERANCE * abs(expected), half_unit) + 1e-9
    return abs(number - expected) <= tolerance
