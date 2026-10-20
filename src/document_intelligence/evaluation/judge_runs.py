"""Passes du juge par rubrique, persistées après chaque évaluation et reprenables.

Chaque évaluation (juge × passe × item) est écrite sur disque dès qu'elle se
termine, avec toutes ses tentatives (sortie brute, `stop_reason`, tokens,
erreur de validation). Une erreur d'API (crédit, quota, réseau) n'efface donc
rien : le fichier contient tout ce qui a été obtenu, et `resume` saute les
évaluations déjà réussies pour ne pas les repayer.

Statuts : `ok` (verdict valide), `echec` (aucun verdict valide : aucune note
n'est attribuée, l'évaluation compte dans le taux d'échec), `interrompu`
(erreur d'API). Une évaluation `echec` ou `interrompu` est rejouée à la
reprise ; l'ancienne version est conservée dans `superseded`.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from document_intelligence.evaluation.annotation import AnnotationSample
from document_intelligence.evaluation.rubric_judge import (
    CallUsage,
    JudgeAttempt,
    RubricJudge,
    RubricJudgeAPIError,
    RubricJudgeError,
)

Status = Literal["ok", "echec", "interrompu"]


def record_key(judge: str, pass_number: int, item_id: str) -> str:
    return f"{judge}|{pass_number}|{item_id}"


class EvaluationRecord(BaseModel):
    """Une évaluation du juge et toutes ses tentatives."""

    judge: str
    pass_number: int
    item_id: str
    status: Status
    temperature: float | None
    scores: dict[str, int] | None = None
    justifications: dict[str, str] | None = None
    error: str | None = None
    usage: CallUsage
    attempts: list[JudgeAttempt]

    @property
    def key(self) -> str:
        return record_key(self.judge, self.pass_number, self.item_id)


class RunConfig(BaseModel):
    """Configuration d'une mesure ; une reprise exige la même configuration."""

    sample_path: str
    sample_created_at: str
    rubric_id: str
    rubric_version: str
    judges: list[str]
    passes: int


class RunStore(BaseModel):
    """État persistant d'une mesure : évaluations finales et versions rejouées."""

    config: RunConfig
    evaluations: dict[str, EvaluationRecord] = Field(default_factory=dict)
    superseded: list[EvaluationRecord] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        """Écriture atomique : un arrêt brutal laisse l'ancien fichier intact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "RunStore":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def records_for(self, judge: str) -> list[EvaluationRecord]:
        return [r for r in self.evaluations.values() if r.judge == judge]


def open_store(path: Path, config: RunConfig, *, resume: bool) -> RunStore:
    """Nouvel état, ou état existant si `resume`.

    Raises:
        FileExistsError: résultats existants sans `resume` (rien n'est écrasé).
        FileNotFoundError: `resume` sans résultats à reprendre.
        ValueError: `resume` avec une configuration différente.
    """
    if not resume:
        if path.exists():
            raise FileExistsError(
                f"{path} contient déjà des résultats payés : relancez avec --resume, "
                "ou déplacez ce fichier pour repartir de zéro."
            )
        return RunStore(config=config)
    if not path.exists():
        raise FileNotFoundError(f"--resume : aucun résultat à reprendre dans {path}.")
    store = RunStore.load(path)
    if store.config != config:
        raise ValueError(
            "--resume : configuration différente de la mesure en cours.\n"
            f"  fichier : {store.config.model_dump()}\n"
            f"  demandée : {config.model_dump()}"
        )
    return store


def run_passes(
    judge: RubricJudge,
    sample: AnnotationSample,
    passes: int,
    store: RunStore,
    path: Path,
    progress: Callable[[str], None] = print,
) -> None:
    """Exécute les passes de `judge`, en sauvegardant après chaque évaluation.

    Les évaluations déjà `ok` sont sautées. Une erreur d'API est consignée
    (`interrompu`), l'état est sauvegardé, puis l'erreur est propagée.

    Raises:
        RubricJudgeAPIError: erreur d'API ; tout ce qui précède est sur disque.
    """
    for number in range(1, passes + 1):
        for position, item in enumerate(sample.items, start=1):
            key = record_key(judge.model, number, item.id)
            label = f"{judge.model} passe {number}/{passes} [{position}/{len(sample.items)}] {item.id}"
            existing = store.evaluations.get(key)
            if existing is not None and existing.status == "ok":
                progress(f"  {label} déjà réussie, sautée")
                continue
            if existing is not None:
                store.superseded.append(existing)

            def make(
                status: Status,
                usage: CallUsage,
                attempts: list[JudgeAttempt],
                *,
                error: str | None = None,
                scores: dict[str, int] | None = None,
                justifications: dict[str, str] | None = None,
                number: int = number,
                item_id: str = item.id,
            ) -> EvaluationRecord:
                return EvaluationRecord(
                    judge=judge.model,
                    pass_number=number,
                    item_id=item_id,
                    status=status,
                    temperature=judge.temperature,
                    scores=scores,
                    justifications=justifications,
                    error=error,
                    usage=usage,
                    attempts=attempts,
                )

            try:
                call = judge.judge(item.judge_input())
            except RubricJudgeError as exc:
                record = make("echec", exc.usage, exc.attempts, error=str(exc))
            except RubricJudgeAPIError as exc:
                store.evaluations[key] = make(
                    "interrompu", exc.usage, exc.attempts, error=str(exc)
                )
                store.save(path)
                progress(f"  {label} INTERROMPU (erreur d'API)")
                raise
            else:
                record = make(
                    "ok",
                    call.usage,
                    call.attempts,
                    scores=call.scores,
                    justifications=call.justifications,
                )
            store.evaluations[key] = record
            store.save(path)
            progress(f"  {label} {'ok' if record.status == 'ok' else 'ÉCHEC'}")


class FailureSummary(BaseModel):
    """Bilan des évaluations d'un juge ; le taux d'échec n'exclut rien."""

    evaluations: int
    ok: int
    failures: int
    interrupted: int
    failure_rate: float | None
    replayed_failures: int
    replayed_interruptions: int


def failure_summary(store: RunStore, judge: str) -> FailureSummary:
    """`failure_rate` = échecs / (réussites + échecs), interruptions exclues."""
    records = store.records_for(judge)
    ok = sum(r.status == "ok" for r in records)
    failures = sum(r.status == "echec" for r in records)
    replayed = [r for r in store.superseded if r.judge == judge]
    return FailureSummary(
        evaluations=len(records),
        ok=ok,
        failures=failures,
        interrupted=sum(r.status == "interrompu" for r in records),
        failure_rate=failures / (ok + failures) if ok + failures else None,
        replayed_failures=sum(r.status == "echec" for r in replayed),
        replayed_interruptions=sum(r.status == "interrompu" for r in replayed),
    )
