"""Exécute l'agent de recherche (tool calling natif) sur des questions réelles.

Indexe le corpus de démonstration (fixtures + corpus étendu) dans un dossier
temporaire, puis pose à l'agent un échantillon reproductible du jeu de test
(`--sample`, `--seed`). Chaque appel d'outil est journalisé ; le rapport JSON
(`--output`) contient, par question, les outils choisis, leurs arguments, la
réponse et les tokens, plus le coût total (agent + synthèse de ask_question).

    uv run python scripts/research_agent_demo.py --sample 6 --output agent_run.json

Appels payants : agent (rôle `agent`) et synthèse RAG (rôle `synthesis`).
"""

import argparse
import json
import random
import tempfile
from pathlib import Path

import anthropic

from document_intelligence.core.config import settings
from document_intelligence.core.logging import configure_logging
from document_intelligence.evaluation.corpus import index_demo_corpus
from document_intelligence.evaluation.runner import load_dataset
from document_intelligence.orchestration.backends import LangGraphBackend
from document_intelligence.orchestration.cost import CostLedger
from document_intelligence.orchestration.research_agent import ResearchAgent
from document_intelligence.orchestration.tiering import (
    LLMRole,
    build_llm_client,
    model_for,
)
from document_intelligence.vectorization.hybrid_store import HybridStore

GOLDEN = Path("src/document_intelligence/evaluation/golden")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configure_logging()

    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY absente.")
    examples = [
        ex
        for name in ("fixtures_qa.jsonl", "extended_qa.jsonl")
        for ex in load_dataset(GOLDEN / name)
    ]
    picked = sorted(random.Random(args.seed).sample(range(len(examples)), args.sample))

    ledger = CostLedger()
    model = model_for(LLMRole.SYNTHESIS)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = HybridStore(index_demo_corpus(Path(tmp)))
        backend = LangGraphBackend(
            store, build_llm_client(LLMRole.SYNTHESIS, ledger=ledger)
        )
        agent = ResearchAgent(
            anthropic.Anthropic(api_key=settings.anthropic_api_key),
            model,
            store,
            backend,
            max_steps=args.max_steps,
            ledger=ledger,
        )
        runs = []
        for index in picked:
            example = examples[index]
            result = agent.run(example.question)
            runs.append(
                {
                    "index": index,
                    "answerable": example.answerable,
                    "expected_substrings": example.relevant_substrings,
                    **result.model_dump(),
                }
            )
            tools_used = [c.name for c in result.tool_calls]
            print(f"\n[{index}] {example.question}\n  outils : {tools_used}")
            print(f"  réponse : {result.answer[:300]}")

    summary = ledger.summary(reference_model=model)
    report = {"model": model, "seed": args.seed, "runs": runs, "cost": summary}
    print("\ncoût :", json.dumps(summary, indent=2))
    if args.output:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
