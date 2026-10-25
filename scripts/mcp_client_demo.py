"""Client MCP de démonstration : lance le serveur en stdio et appelle ses outils.

C'est ce que fait un client comme Claude Desktop ou Claude Code : démarrer
`python -m document_intelligence.mcp_server` en sous-processus et lui parler
en JSON-RPC sur stdin/stdout.

    # index de démonstration (fixtures + corpus étendu) dans un dossier temporaire
    uv run python scripts/mcp_client_demo.py --demo-corpus \
        --question "Quel est le montant de la redevance annuelle ?"

    # ou sur un index Chroma existant
    uv run python scripts/mcp_client_demo.py --chroma-dir .chroma --question "..."

`ask_question` appelle réellement le fournisseur LLM configuré (`.env`) : coût
d'un appel de synthèse. Sans clé, l'outil renvoie une erreur explicite.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from document_intelligence.evaluation.corpus import index_demo_corpus


def _show(label: str, result: Any) -> None:
    payload = result.structuredContent or {"text": result.content[0].text}
    print(f"\n=== {label} (isError={result.isError})")
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:3000])


async def run(persist: Path, question: str, search: str) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "document_intelligence.mcp_server"],
        env={**os.environ, "CHROMA_PERSIST_DIRECTORY": str(persist)},
    )
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        init = await session.initialize()
        tools = (await session.list_tools()).tools
        print(f"serveur : {init.serverInfo.name} ; outils : {[t.name for t in tools]}")

        _show("list_documents", await session.call_tool("list_documents", {}))
        _show(
            "search_documents",
            await session.call_tool("search_documents", {"query": search, "k": 3}),
        )
        start = time.perf_counter()
        answer = await session.call_tool("ask_question", {"question": question})
        elapsed = time.perf_counter() - start
        _show("ask_question", answer)
        print(f"\nlatence ask_question (aller-retour MCP) : {elapsed:.2f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chroma-dir", type=Path, default=Path(".chroma"))
    parser.add_argument("--demo-corpus", action="store_true")
    parser.add_argument(
        "--question", default="Quel est le montant de la redevance annuelle ?"
    )
    parser.add_argument("--search", default="redevance annuelle du contrat de licence")
    args = parser.parse_args()

    # ignore_cleanup_errors : sous Windows, Chroma garde ses fichiers ouverts
    # jusqu'à la fin du processus.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        if args.demo_corpus:
            index_demo_corpus(Path(tmp))
            persist = Path(tmp) / "chroma"
        else:
            persist = args.chroma_dir
        anyio.run(run, persist, args.question, args.search)


if __name__ == "__main__":
    main()
