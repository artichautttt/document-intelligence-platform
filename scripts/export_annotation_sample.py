"""Exporte l'échantillon d'annotation manuelle de l'évaluation par rubrique.

1. Tire un échantillon stratifié (graine fixe) parmi les 63 questions du jeu de
   test : toutes les questions sans réponse possible, plus des questions
   répondables au prorata de chaque fichier.
2. Exécute le pipeline RAG réel sur ces questions (ingestion des documents
   d'évaluation, retrieval, synthèse par `LLM_SYNTHESIS_MODEL`) et conserve
   réponse, chunks récupérés et citations.
3. Écrit `sample.json` (référence du juge) et `annotations.csv` (à remplir à la
   main, sans aucune note de juge).

Appelle l'API du fournisseur LLM pour la synthèse (une requête par question).
Refuse d'écraser un CSV existant, qui peut contenir des annotations.

    uv run python scripts/export_annotation_sample.py
"""

import argparse
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from document_intelligence.chunking.service import chunk_document
from document_intelligence.core.config import settings
from document_intelligence.core.logging import configure_logging
from document_intelligence.evaluation.annotation import (
    AnnotationChunk,
    AnnotationItem,
    AnnotationSample,
    load_golden_examples,
    stratified_sample,
    stratum,
    write_annotation_csv,
    write_sample,
)
from document_intelligence.evaluation.corpus import build_extended_corpus
from document_intelligence.evaluation.rubric_judge import (
    DEFAULT_RUBRIC_PATH,
    load_rubric,
)
from document_intelligence.ingestion.service import ingest
from document_intelligence.orchestration.cost import CostLedger
from document_intelligence.orchestration.graph import answer_query
from document_intelligence.orchestration.tiering import (
    LLMRole,
    build_llm_client,
    model_for,
    reference_model,
)
from document_intelligence.vectorization.chroma_store import ChromaStore
from document_intelligence.vectorization.hybrid_store import HybridStore
from document_intelligence.vectorization.store import VectorStore

_GOLDEN_DIR = Path("src/document_intelligence/evaluation/golden")
_DATASETS = [_GOLDEN_DIR / "fixtures_qa.jsonl", _GOLDEN_DIR / "extended_qa.jsonl"]
_FIXTURES = Path("tests/fixtures/documents")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=30, help="Taille de l'échantillon.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--retrieval-mode", choices=["dense", "hybrid"], default=settings.retrieval_mode
    )
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC_PATH)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("annotations/rag_answer_v1")
    )
    return parser.parse_args()


def _build_store(directory: Path, retrieval_mode: str) -> VectorStore:
    """Indexe les documents d'évaluation, comme `python -m document_intelligence.evaluation`."""
    build_extended_corpus(directory / "corpus")
    chroma = ChromaStore(persist_directory=str(directory / "chroma"))
    for folder in (_FIXTURES, directory / "corpus"):
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in {".pdf", ".docx"}:
                chroma.add_chunks(chunk_document(ingest(path)))
    return HybridStore(chroma) if retrieval_mode == "hybrid" else chroma


def main() -> int:
    args = _parse_args()
    configure_logging()
    csv_path = args.out_dir / "annotations.csv"
    json_path = args.out_dir / "sample.json"
    if csv_path.exists():
        print(
            f"{csv_path} existe déjà et peut contenir des annotations : rien n'est "
            "écrasé. Déplacez-le ou choisissez un autre --out-dir."
        )
        return 1

    rubric = load_rubric(args.rubric)
    population = load_golden_examples(_DATASETS)
    selected = stratified_sample(population, n=args.n, seed=args.seed)
    ledger = CostLedger()
    llm = build_llm_client(LLMRole.SYNTHESIS, cache=None, ledger=ledger)

    items: list[AnnotationItem] = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = _build_store(Path(tmp), args.retrieval_mode)
        for position, sourced in enumerate(selected, start=1):
            question = sourced.example.question
            retrieved = store.query(question, k=args.k)
            if [r.chunk_id for r in store.query(question, k=args.k)] != [
                r.chunk_id for r in retrieved
            ]:
                raise RuntimeError(f"retrieval non déterministe pour {sourced.id}")
            answer = answer_query(question, store=store, llm=llm, k=args.k)
            cited = {c.chunk_id for c in answer.citations}
            unknown = cited - {r.chunk_id for r in retrieved}
            if unknown:
                raise RuntimeError(f"{sourced.id} : citations hors contexte {unknown}")
            items.append(
                AnnotationItem(
                    id=sourced.id,
                    source_file=sourced.source_file,
                    answerable=sourced.example.answerable,
                    question=question,
                    answer=answer.answer,
                    sufficient_context=answer.sufficient_context,
                    chunks=[
                        AnnotationChunk(
                            chunk_id=r.chunk_id,
                            source=Path(r.provenance.source_path).name,
                            text=r.text,
                            cited=r.chunk_id in cited,
                        )
                        for r in retrieved
                    ],
                )
            )
            print(f"[{position}/{len(selected)}] {sourced.id}")

    strata: dict[str, int] = {}
    for sourced in selected:
        strata[stratum(sourced)] = strata.get(stratum(sourced), 0) + 1
    sample = AnnotationSample(
        rubric_id=rubric.id,
        rubric_version=rubric.version,
        seed=args.seed,
        population_size=len(population),
        strata=dict(sorted(strata.items())),
        synthesis_model=model_for(LLMRole.SYNTHESIS),
        k=args.k,
        retrieval_mode=args.retrieval_mode,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        items=items,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_sample(sample, json_path)
    write_annotation_csv(sample, rubric, csv_path)

    cost = ledger.summary(reference_model=reference_model())
    print(f"\nÉchantillon : N={len(items)} sur {len(population)} (graine {args.seed})")
    for key, count in sample.strata.items():
        print(f"  {key:<40} {count}")
    print(
        f"Synthèse : {sample.synthesis_model}, {int(cost['llm_calls'])} appel(s), "
        f"coût {cost['cost_actual_usd']:.4f} $"
    )
    print(f"Référence du juge : {json_path}")
    print(f"CSV à annoter      : {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
