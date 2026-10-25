"""Bout en bout : un client MCP lance le vrai serveur en sous-processus (stdio).

Le serveur est démarré comme le ferait Claude Desktop ou Claude Code
(`python -m document_intelligence.mcp_server`), sur un vrai ChromaDB contenant
les trois documents de `tests/fixtures/documents/` (embeddings ONNX locaux).
Aucune clé LLM : `ask_question` et `ask_data` doivent répondre par une erreur
explicite ; les outils sans LLM (recherche, liste, `run_sql` sur la base taxi
versionnée) fonctionnent réellement.

Le sous-processus tourne dans un répertoire vide, pour ne pas lire le `.env`
du dépôt (une clé locale ferait un vrai appel payant).
"""

import os
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from document_intelligence.chunking.service import chunk_document
from document_intelligence.ingestion.service import ingest
from document_intelligence.vectorization.chroma_store import ChromaStore

FIXTURES = ["financial_report.pdf", "legal_contract.docx", "audit_report.docx"]


@pytest.fixture
def indexed_chroma(fixtures_dir: Path, tmp_path: Path) -> Path:
    persist = tmp_path / "chroma"
    store = ChromaStore(persist_directory=str(persist))
    for name in FIXTURES:
        store.add_chunks(chunk_document(ingest(fixtures_dir / name)))
    return persist


def _server_params(persist: Path, workdir: Path) -> StdioServerParameters:
    env = {
        key: value for key, value in os.environ.items() if not key.endswith("_API_KEY")
    }
    env.update(
        {
            "CHROMA_PERSIST_DIRECTORY": str(persist),
            "VECTOR_STORE_BACKEND": "chroma",
            "RETRIEVAL_MODE": "hybrid",
            "LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "",
            "ANSWER_VERIFICATION": "none",
        }
    )
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "document_intelligence.mcp_server"],
        env=env,
        cwd=str(workdir),
    )


def test_real_client_calls_the_tools_over_stdio(
    indexed_chroma: Path, tmp_path: Path
) -> None:
    workdir = tmp_path / "empty_workdir"
    workdir.mkdir()

    async def run() -> dict[str, Any]:
        async with (
            stdio_client(_server_params(indexed_chroma, workdir)) as (read, write),
            ClientSession(read, write) as session,
        ):
            init = await session.initialize()
            return {
                "server": init.serverInfo.name,
                "tools": sorted(t.name for t in (await session.list_tools()).tools),
                "documents": await session.call_tool("list_documents", {}),
                "search": await session.call_tool(
                    "search_documents",
                    {"query": "résiliation du contrat en cas de manquement", "k": 3},
                ),
                "ask": await session.call_tool(
                    "ask_question", {"question": "Quel est le chiffre d'affaires ?"}
                ),
                "sql": await session.call_tool(
                    "run_sql", {"sql": "SELECT SUM(trip_count) FROM hourly_trips"}
                ),
                "sql_write": await session.call_tool(
                    "run_sql", {"sql": "DELETE FROM zones"}
                ),
                "ask_data": await session.call_tool(
                    "ask_data", {"question": "Combien de trajets en janvier ?"}
                ),
            }

    out = anyio.run(run)

    assert out["server"] == "document-intelligence"
    assert out["tools"] == [
        "ask_data",
        "ask_question",
        "describe_tables",
        "list_documents",
        "run_sql",
        "search_documents",
    ]

    documents = out["documents"].structuredContent["documents"]
    assert sorted(Path(d["source_path"]).name for d in documents) == sorted(FIXTURES)
    assert all(d["chunk_count"] > 0 for d in documents)

    hits = out["search"].structuredContent["hits"]
    assert len(hits) == 3
    assert Path(hits[0]["source_path"]).name == "legal_contract.docx"

    assert out["ask"].isError
    assert "Aucun fournisseur LLM" in out["ask"].content[0].text

    assert out["sql"].structuredContent["rows"] == [[3330486]]
    assert out["sql_write"].isError
    assert "lecture seule" in out["sql_write"].content[0].text
    assert out["ask_data"].isError
    assert "ask_data est indisponible" in out["ask_data"].content[0].text
