"""CLI d'évaluation : `python -m document_intelligence.evaluation`.

Ingère les documents dans un vector store éphémère, exécute le jeu de test et
affiche les métriques. Sans clé Anthropic (ou avec `--retrieval-only`), seul
le retrieval est évalué : aucune requête réseau vers un LLM.

Par défaut, l'évaluation porte sur les fixtures de `tests/fixtures/documents/`
et sur le corpus synthétique étendu (`evaluation/corpus.py`), avec les deux
jeux de questions de `evaluation/golden/`.
"""

import argparse
import random
import sys
import tempfile
from pathlib import Path

from document_intelligence.chunking.service import chunk_document
from document_intelligence.core.config import settings
from document_intelligence.core.logging import configure_logging
from document_intelligence.evaluation.corpus import build_extended_corpus
from document_intelligence.evaluation.runner import (
    AnswerFn,
    dump_report,
    evaluate,
    format_report,
    load_dataset,
)
from document_intelligence.evaluation.tracking import log_evaluation_run
from document_intelligence.ingestion.service import ingest
from document_intelligence.orchestration.cost import CostLedger
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.orchestration.tiering import (
    LLMRole,
    active_provider,
    build_llm_client,
    build_response_cache,
    model_for,
    provider_api_key,
    reference_model,
)
from document_intelligence.orchestration.verification import AnswerVerifier
from document_intelligence.vectorization.chroma_store import ChromaStore
from document_intelligence.vectorization.hybrid_store import HybridStore
from document_intelligence.vectorization.store import VectorStore

_GOLDEN_DIR = Path(__file__).parent / "golden"
_DEFAULT_DATASETS = [
    _GOLDEN_DIR / "fixtures_qa.jsonl",
    _GOLDEN_DIR / "extended_qa.jsonl",
]
_DEFAULT_FIXTURES = Path("tests/fixtures/documents")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évaluation du pipeline RAG.")
    parser.add_argument("--dataset", type=Path, nargs="+", default=_DEFAULT_DATASETS)
    parser.add_argument(
        "--documents",
        type=Path,
        nargs="+",
        help="Dossiers de PDF/DOCX à ingérer (défaut : fixtures + corpus étendu).",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "hybrid"],
        default=settings.retrieval_mode,
    )
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["langgraph", "langchain"],
        default=settings.rag_backend,
        help="Backend de réponse : graphe LangGraph (défaut) ou chaîne LangChain.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Évalue un sous-ensemble tiré au hasard (graine --seed) de N questions.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Insère l'agent de vérification CrewAI dans le pipeline (groupe `agents`).",
    )
    parser.add_argument(
        "--calibrate-verifier",
        action="store_true",
        help="Avec --verify : soumet aussi une copie faussée de chaque réponse.",
    )
    parser.add_argument(
        "--max-regenerations",
        type=int,
        default=settings.verification_max_regenerations,
    )
    parser.add_argument(
        "--local-model",
        help=(
            "Synthèse par un modèle Hugging Face local (ex. Qwen/Qwen2.5-0.5B-Instruct) "
            "au lieu du fournisseur ; le juge reste celui du fournisseur. "
            "Dépendances : finetune/requirements.txt."
        ),
    )
    parser.add_argument("--local-adapter", help="Adaptateur LoRA à fusionner.")
    parser.add_argument("--output", type=Path, help="Écrit le rapport JSON détaillé.")
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        help="Code de sortie 1 si le hit rate est inférieur (garde-fou CI).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging()

    llm: LLMClient | None = None
    judge: LLMClient | None = None
    ledger = CostLedger()
    if not args.retrieval_only and provider_api_key():
        # Cache en mémoire partagé par les deux rôles, propre à ce run.
        cache = build_response_cache()
        llm = build_llm_client(
            LLMRole.SYNTHESIS,
            cache=cache,
            ledger=ledger,
        )
        judge = build_llm_client(
            LLMRole.JUDGE,
            cache=cache,
            ledger=ledger,
        )
        if args.local_model:
            from document_intelligence.orchestration.hf_local_client import (
                HFLocalLLMClient,
            )

            llm = HFLocalLLMClient(args.local_model, adapter=args.local_adapter)
    verifier: AnswerVerifier | None = None
    if llm is not None and args.verify:
        from document_intelligence.orchestration.crew_verifier import CrewAIVerifier

        verifier = CrewAIVerifier(
            build_llm_client(LLMRole.VERIFIER, cache=cache, ledger=ledger)
        )
    elif not args.retrieval_only and not llm:
        print(
            f"Clé d'API absente pour LLM_PROVIDER={active_provider()} : "
            "évaluation du retrieval uniquement."
        )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        directories = args.documents
        if directories is None:
            build_extended_corpus(Path(tmp) / "corpus")
            directories = [_DEFAULT_FIXTURES, Path(tmp) / "corpus"]

        chroma = ChromaStore(persist_directory=str(Path(tmp) / "chroma"))
        for directory in directories:
            for path in sorted(directory.iterdir()):
                if path.suffix.lower() in {".pdf", ".docx"}:
                    chroma.add_chunks(chunk_document(ingest(path)))
        store: VectorStore = (
            HybridStore(chroma) if args.retrieval_mode == "hybrid" else chroma
        )
        answer_fn: AnswerFn | None = None
        if llm is not None and args.backend == "langchain":
            if verifier is not None or args.local_model:
                raise SystemExit(
                    "--backend langchain : incompatible avec --verify et --local-model."
                )
            from document_intelligence.orchestration.langchain_backend import (
                build_langchain_rag,
            )

            # Même journal de coût que le juge : coûts comparables entre backends.
            answer_fn = build_langchain_rag(store, ledger=ledger).answer

        examples = [ex for dataset in args.dataset for ex in load_dataset(dataset)]
        if args.sample is not None and args.sample < len(examples):
            # Sous-ensemble reproductible, dans l'ordre d'origine du jeu de test.
            kept = sorted(
                random.Random(args.seed).sample(range(len(examples)), args.sample)
            )
            examples = [examples[i] for i in kept]
        report = evaluate(
            examples,
            store,
            k=args.k,
            llm=llm,
            judge=judge,
            verifier=verifier,
            max_regenerations=args.max_regenerations,
            calibrate_verifier=args.calibrate_verifier,
            answer_fn=answer_fn,
        )

    if llm is not None:
        report.cost = ledger.summary(reference_model=reference_model())

    print(f"[retrieval={args.retrieval_mode}] [backend={args.backend}]")
    print(format_report(report))
    if report.cost:
        print(f"Coût LLM (référence sans optimisation : {reference_model()})")
        print("\n".join(f"  {k:<32} {v:.6g}" for k, v in report.cost.items()))
    if args.output:
        dump_report(report, args.output)

    log_evaluation_run(
        report,
        retrieval_mode=args.retrieval_mode,
        llm_model=model_for(LLMRole.JUDGE) if llm is not None else None,
        synthesis_model=(
            None
            if llm is None
            else getattr(llm, "model", None)
            if args.local_model
            else model_for(LLMRole.SYNTHESIS)
        ),
        report_path=args.output,
        rag_backend=args.backend,
    )

    if (
        args.min_hit_rate is not None
        and report.aggregate.get("hit_rate", 0.0) < args.min_hit_rate
    ):
        print(f"hit_rate < {args.min_hit_rate} : échec du garde-fou.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
