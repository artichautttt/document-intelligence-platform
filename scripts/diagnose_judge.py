"""Rejoue le juge par rubrique sur quelques items et affiche chaque tentative en entier.

Diagnostic des verdicts invalides : pour chaque tentative, la sortie brute
complète du modèle, `stop_reason`, les tokens et l'erreur de validation exacte.
Le prompt, le schéma et la rubrique sont ceux de la mesure, inchangés. Rien
n'est écrit dans `reliability_raw.json` ; les tentatives sont enregistrées dans
un fichier de diagnostic séparé (`--output`).

Appelle l'API Anthropic : une ou deux requêtes par item.

    uv run python scripts/diagnose_judge.py --judge claude-sonnet-5 \
        --ids fixtures_qa#09 fixtures_qa#11 extended_qa#04 extended_qa#05 extended_qa#10
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from document_intelligence.core.logging import configure_logging
from document_intelligence.evaluation.annotation import load_sample
from document_intelligence.evaluation.rubric_judge import (
    DEFAULT_RUBRIC_PATH,
    JudgeAttempt,
    RubricJudgeAPIError,
    RubricJudgeError,
    build_rubric_judge,
    load_rubric,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    out_dir = Path("annotations/rag_answer_v1")
    parser.add_argument("--sample", type=Path, default=out_dir / "sample.json")
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC_PATH)
    parser.add_argument("--judge", default="claude-sonnet-5")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path, help="Défaut : diagnostic_<horodatage>.json"
    )
    return parser.parse_args()


def _show(item_id: str, status: str, attempts: list[JudgeAttempt]) -> None:
    print(f"\n{'=' * 80}\n{item_id} — {status} ({len(attempts)} tentative(s))")
    for attempt in attempts:
        print(
            f"\n--- tentative {attempt.attempt} : stop_reason={attempt.stop_reason}, "
            f"tokens d'entrée={attempt.input_tokens}, "
            f"tokens de sortie={attempt.output_tokens}"
        )
        print(f"erreur de validation : {attempt.validation_error or 'aucune'}")
        print("sortie brute :")
        print(attempt.raw_output)


def main() -> int:
    args = _parse_args()
    configure_logging()
    rubric = load_rubric(args.rubric)
    sample = load_sample(args.sample)
    items = {item.id: item for item in sample.items}
    unknown = [i for i in args.ids if i not in items]
    if unknown:
        print(f"Ids absents de l'échantillon : {', '.join(unknown)}")
        return 1

    output = args.output or args.sample.parent / (
        f"diagnostic_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    judge = build_rubric_judge(args.judge, rubric=rubric)
    results: list[dict[str, Any]] = []
    exit_code = 0
    for item_id in args.ids:
        try:
            call = judge.judge(items[item_id].judge_input())
        except RubricJudgeError as exc:
            status, attempts, scores = "échec", exc.attempts, None
        except RubricJudgeAPIError as exc:
            status, attempts, scores = "interrompu (erreur d'API)", exc.attempts, None
            print(f"\nErreur d'API : {exc}")
            exit_code = 2
        else:
            status, attempts, scores = "verdict valide", call.attempts, call.scores
        _show(item_id, status, attempts)
        results.append(
            {
                "id": item_id,
                "status": status,
                "scores": scores,
                "attempts": [a.model_dump() for a in attempts],
            }
        )
        if exit_code:
            break

    output.write_text(
        json.dumps(
            {
                "judge": args.judge,
                "temperature": judge.temperature,
                "rubric": f"{rubric.id} {rubric.version}",
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nDiagnostic enregistré : {output}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
