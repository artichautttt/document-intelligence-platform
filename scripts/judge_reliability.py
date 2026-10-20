"""Mesure la stabilité et la fiabilité du juge par rubrique face aux annotations manuelles.

À lancer seulement une fois `annotations.csv` entièrement rempli : le script
refuse de démarrer (sans aucun appel d'API) si une note manque ou est invalide.

Pour chaque modèle juge (par défaut `claude-sonnet-5` et `claude-haiku-4-5`) :

- **passes** : le juge note chaque réponse de l'échantillon `--passes` fois (3) ;
- **stabilité** : accord entre passes, par critère — part des items notés à
  l'identique par toutes les passes, et kappa pondéré quadratique moyen sur
  les paires de passes ;
- **fiabilité** : note médiane des passes vs notes manuelles, par critère —
  accord exact, kappa pondéré quadratique, Spearman, avec IC bootstrap 95 % ;
- **taux d'échec** (verdicts invalides, jamais remplacés par une note) ;
- **coût et latence** moyens par évaluation ;
- un run MLflow par juge (expérience `rag-judge-reliability`), avec le CSV
  annoté, les résultats bruts et le rapport JSON en artefacts.

**Persistance et reprise** : chaque évaluation est écrite dans
`reliability_raw.json` dès qu'elle se termine, avec ses tentatives (sortie
brute, `stop_reason`, tokens, erreur de validation). Après une erreur d'API
(crédit, quota, réseau), relancer avec `--resume` saute les évaluations déjà
réussies et rejoue les autres. Sans `--resume`, un fichier existant n'est
jamais écrasé.

Une métrique non définie (variance nulle, par exemple) est affichée « non
calculable » et vaut `null` dans le rapport.

    uv run python scripts/judge_reliability.py
    uv run python scripts/judge_reliability.py --resume
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from document_intelligence.core.logging import configure_logging
from document_intelligence.evaluation.annotation import (
    AnnotationError,
    load_sample,
    read_annotations,
)
from document_intelligence.evaluation.judge_runs import (
    RunConfig,
    RunStore,
    failure_summary,
    open_store,
    record_key,
    run_passes,
)
from document_intelligence.evaluation.reliability import (
    BootstrapInterval,
    Metric,
    all_passes_agree,
    bootstrap_ci,
    exact_agreement,
    mean_if_all_defined,
    median_score,
    pairwise,
    quadratic_weighted_kappa,
    spearman,
)
from document_intelligence.evaluation.rubric_judge import (
    DEFAULT_RUBRIC_PATH,
    Rubric,
    RubricJudgeAPIError,
    build_rubric_judge,
    judge_temperature,
    load_rubric,
)
from document_intelligence.evaluation.tracking import log_reliability_run

NOT_COMPUTABLE = "non calculable"
EXIT_API_ERROR = 2
_AGREEMENT_METRICS: dict[str, Metric] = {
    "exact": exact_agreement,
    "kappa_quadratic": quadratic_weighted_kappa,
    "spearman": spearman,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    out_dir = Path("annotations/rag_answer_v1")
    parser.add_argument("--sample", type=Path, default=out_dir / "sample.json")
    parser.add_argument("--annotations", type=Path, default=out_dir / "annotations.csv")
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC_PATH)
    parser.add_argument(
        "--judges", nargs="+", default=["claude-sonnet-5", "claude-haiku-4-5"]
    )
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--raw",
        type=Path,
        default=out_dir / "reliability_raw.json",
        help="Résultats bruts, écrits après chaque évaluation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprend --raw : saute les évaluations réussies, rejoue les autres.",
    )
    parser.add_argument(
        "--output", type=Path, default=out_dir / "reliability_report.json"
    )
    args = parser.parse_args()
    if args.passes < 3 or args.passes % 2 == 0:
        parser.error(
            "--passes doit être impair et >= 3 (médiane et accord entre passes)"
        )
    return args


def _interval(ci: BootstrapInterval | None) -> dict[str, Any] | None:
    if ci is None:
        return None
    return {
        "low": ci.low,
        "high": ci.high,
        "level": ci.level,
        "resamples": ci.resamples,
        "defined_resamples": ci.defined,
    }


def _criterion_results(
    rubric: Rubric,
    ids: Sequence[str],
    passes_scores: list[dict[str, dict[str, int] | None]],
    human: dict[str, dict[str, int]],
    resamples: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    """Stabilité et fiabilité par critère, sur les items valides à toutes les passes."""
    complete = [i for i in ids if all(p.get(i) is not None for p in passes_scores)]
    results: dict[str, Any] = {}
    for criterion in rubric.criterion_ids:
        passes = [
            [p[i][criterion] for i in complete]  # type: ignore[index]
            for p in passes_scores
        ]
        kappas = pairwise(quadratic_weighted_kappa, passes)
        judged = [median_score([p[k] for p in passes]) for k in range(len(complete))]
        manual = [human[i][criterion] for i in complete]
        reliability: dict[str, Any] = {"n": len(complete)}
        for name, metric in _AGREEMENT_METRICS.items():
            reliability[name] = {
                "value": metric(judged, manual),
                "ci": _interval(
                    bootstrap_ci(judged, manual, metric, resamples=resamples, seed=seed)
                ),
            }
        results[criterion] = {
            "stability": {
                "n": len(complete),
                "exact_all_passes": all_passes_agree(passes),
                "kappa_quadratic_pairs": kappas,
                "kappa_quadratic_mean": mean_if_all_defined(kappas),
            },
            "reliability": reliability,
            "distribution": {
                "human": dict(sorted(Counter(manual).items())),
                "judge_median": dict(sorted(Counter(judged).items())),
            },
        }
    return complete, results


def _usage_summary(store: RunStore, judge: str) -> dict[str, Any]:
    """Moyennes sur les évaluations finales ; le total inclut les rejouées (payées)."""
    final = store.records_for(judge)
    replayed = [r for r in store.superseded if r.judge == judge]
    n = len(final)

    def mean(values: list[float]) -> float | None:
        return sum(values) / n if n else None

    return {
        "evaluations": n,
        "api_calls": sum(r.usage.api_calls for r in final + replayed),
        "cost_total_usd": sum(r.usage.cost_usd for r in final + replayed),
        "cost_mean_usd": mean([r.usage.cost_usd for r in final]),
        "latency_mean_ms": mean([r.usage.latency_ms for r in final]),
        "input_tokens_mean": mean([float(r.usage.input_tokens) for r in final]),
        "output_tokens_mean": mean([float(r.usage.output_tokens) for r in final]),
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return NOT_COMPUTABLE if value is None else f"{value:.{digits}f}"


def _fmt_ci(entry: dict[str, Any]) -> str:
    ci = entry["ci"]
    interval = NOT_COMPUTABLE if ci is None else f"[{ci['low']:.2f} ; {ci['high']:.2f}]"
    return f"{_fmt(entry['value'])} {interval}"


def _print_judge(judge: dict[str, Any], passes: int, resamples: int) -> None:
    usage, failures = judge["usage"], judge["failures"]
    temperature = (
        "non envoyée (refusée par ce modèle)"
        if judge["temperature"] is None
        else f"{judge['temperature']}"
    )
    n = judge["n_complete"]
    print(f"\n=== Juge {judge['model']} — température : {temperature} ===")
    print(
        f"Évaluations : {failures['evaluations']} ({passes} passes × "
        f"{judge['n_sample']} réponses) — réussies {failures['ok']}, "
        f"échecs {failures['failures']} (taux d'échec "
        f"{_fmt(failures['failure_rate'])}), interrompues {failures['interrupted']}"
    )
    print(
        f"Rejouées après reprise : {failures['replayed_failures']} échec(s), "
        f"{failures['replayed_interruptions']} interruption(s)"
    )
    print(f"Réponses valides aux {passes} passes : N={n}")
    print(
        f"Coût moyen / évaluation : {_fmt(usage['cost_mean_usd'], 5)} $ "
        f"(N={usage['evaluations']}) ; coût total (rejouées comprises) : "
        f"{usage['cost_total_usd']:.4f} $ ; latence moyenne : "
        f"{_fmt(usage['latency_mean_ms'], 0)} ms (N={usage['evaluations']})"
    )
    print(f"\nStabilité entre {passes} passes (N={n})")
    print(f"  {'critère':<24}{'accord exact':>14}{'kappa pondéré moyen':>22}")
    for criterion, result in judge["criteria"].items():
        stability = result["stability"]
        print(
            f"  {criterion:<24}{_fmt(stability['exact_all_passes']):>14}"
            f"{_fmt(stability['kappa_quadratic_mean']):>22}"
        )
    print(
        f"\nFiabilité vs annotations manuelles — médiane des {passes} passes, "
        f"N={n}, IC bootstrap 95 % (B={resamples})"
    )
    print(f"  {'critère':<24}{'accord exact':>26}{'kappa pondéré':>26}{'Spearman':>26}")
    for criterion, result in judge["criteria"].items():
        rel = result["reliability"]
        print(
            f"  {criterion:<24}{_fmt_ci(rel['exact']):>26}"
            f"{_fmt_ci(rel['kappa_quadratic']):>26}{_fmt_ci(rel['spearman']):>26}"
        )
    print(f"\nDistribution des notes (N={n}) — humain | juge (médiane)")
    for criterion, result in judge["criteria"].items():
        dist = result["distribution"]
        print(f"  {criterion:<24}{dist['human']} | {dist['judge_median']}")


def _mlflow_metrics(judge: dict[str, Any]) -> dict[str, float]:
    failures = judge["failures"]
    metrics: dict[str, float] = {
        "n_complete": float(judge["n_complete"]),
        "failures": float(failures["failures"]),
        "replayed_failures": float(failures["replayed_failures"]),
        "cost_total_usd": judge["usage"]["cost_total_usd"],
    }
    optional = {
        "failure_rate": failures["failure_rate"],
        "cost_mean_usd": judge["usage"]["cost_mean_usd"],
        "latency_mean_ms": judge["usage"]["latency_mean_ms"],
    }
    for criterion, result in judge["criteria"].items():
        stability = result["stability"]
        optional[f"stability_exact_{criterion}"] = stability["exact_all_passes"]
        optional[f"stability_kappa_{criterion}"] = stability["kappa_quadratic_mean"]
        for name, entry in result["reliability"].items():
            if name == "n":
                continue
            optional[f"reliability_{name}_{criterion}"] = entry["value"]
            if entry["ci"] is not None:
                optional[f"reliability_{name}_{criterion}_ci_low"] = entry["ci"]["low"]
                optional[f"reliability_{name}_{criterion}_ci_high"] = entry["ci"][
                    "high"
                ]
    metrics |= {k: float(v) for k, v in optional.items() if v is not None}
    return metrics


def main() -> int:
    args = _parse_args()
    configure_logging()
    rubric = load_rubric(args.rubric)
    sample = load_sample(args.sample)
    if (sample.rubric_id, sample.rubric_version) != (rubric.id, rubric.version):
        print(
            f"Rubrique {rubric.id} {rubric.version} ≠ celle de l'échantillon "
            f"({sample.rubric_id} {sample.rubric_version})."
        )
        return 1
    ids = [item.id for item in sample.items]
    try:
        human = read_annotations(args.annotations, rubric, ids)
    except AnnotationError as exc:
        print(f"{exc}\n\nAucun juge n'a été lancé : complétez le CSV puis relancez.")
        return 1
    except OSError as exc:
        print(f"CSV d'annotation illisible : {exc}")
        return 1

    config = RunConfig(
        sample_path=str(args.sample),
        sample_created_at=sample.created_at,
        rubric_id=rubric.id,
        rubric_version=rubric.version,
        judges=list(args.judges),
        passes=args.passes,
    )
    try:
        store = open_store(args.raw, config, resume=args.resume)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    for model in args.judges:
        judge = build_rubric_judge(model, rubric=rubric)
        print(f"Juge {model} : {args.passes} passes sur N={len(ids)} réponses")
        try:
            run_passes(judge, sample, args.passes, store, args.raw)
        except RubricJudgeAPIError as exc:
            done = sum(r.status != "interrompu" for r in store.evaluations.values())
            print(
                f"\nArrêt sur erreur d'API : {exc}\n"
                f"{done} évaluation(s) terminée(s) sont enregistrées dans {args.raw} "
                "(tentatives et sorties brutes comprises).\n"
                "Après correction (crédit, quota, réseau), relancez avec --resume : "
                "les évaluations réussies ne seront pas repayées."
            )
            return EXIT_API_ERROR

    judges: list[dict[str, Any]] = []
    for model in args.judges:
        passes_scores = [
            {
                i: (
                    record.scores
                    if (record := store.evaluations[record_key(model, p, i)]).status
                    == "ok"
                    else None
                )
                for i in ids
            }
            for p in range(1, args.passes + 1)
        ]
        complete, criteria = _criterion_results(
            rubric, ids, passes_scores, human, args.bootstrap, args.bootstrap_seed
        )
        judges.append(
            {
                "model": model,
                "temperature": judge_temperature(model),
                "n_sample": len(ids),
                "n_complete": len(complete),
                "failures": failure_summary(store, model).model_dump(),
                "usage": _usage_summary(store, model),
                "criteria": criteria,
            }
        )

    report = {
        "rubric": {"id": rubric.id, "version": rubric.version},
        "sample": {
            "path": str(args.sample),
            "n": len(ids),
            "seed": sample.seed,
            "population_size": sample.population_size,
            "strata": sample.strata,
            "synthesis_model": sample.synthesis_model,
            "k": sample.k,
            "retrieval_mode": sample.retrieval_mode,
        },
        "annotations": str(args.annotations),
        "raw_results": str(args.raw),
        "passes": args.passes,
        "aggregation": "médiane des passes",
        "failure_rate_definition": "échecs / (réussites + échecs)",
        "bootstrap": {"resamples": args.bootstrap, "seed": args.bootstrap_seed},
        "judges": judges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for judge_report in judges:
        _print_judge(judge_report, args.passes, args.bootstrap)
        traced = log_reliability_run(
            params={
                "judge_model": judge_report["model"],
                "temperature": str(judge_report["temperature"]),
                "rubric_id": rubric.id,
                "rubric_version": rubric.version,
                "n_sample": len(ids),
                "sample_seed": sample.seed,
                "passes": args.passes,
                "bootstrap_resamples": args.bootstrap,
                "bootstrap_seed": args.bootstrap_seed,
                "synthesis_model": sample.synthesis_model,
            },
            metrics=_mlflow_metrics(judge_report),
            artifacts=[args.annotations, args.raw, args.output],
        )
        print(f"  MLflow : {'run enregistré' if traced else 'non tracé'}")
    print(f"\nRapport détaillé : {args.output}\nRésultats bruts : {args.raw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
