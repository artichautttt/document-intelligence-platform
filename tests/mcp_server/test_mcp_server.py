"""Outils appelés par un vrai client MCP (`ClientSession`), connecté en mémoire au serveur."""

from collections.abc import Awaitable, Callable
from typing import Any

import anyio
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

from document_intelligence.mcp_server import build_server
from document_intelligence.orchestration.llm import LLMClient

from .conftest import CitingLLM, InMemoryStore


def _with_client(
    store: InMemoryStore,
    llm: LLMClient | None,
    scenario: Callable[[ClientSession], Awaitable[Any]],
) -> Any:
    async def run() -> Any:
        async with create_connected_server_and_client_session(
            build_server(store, llm)
        ) as client:
            return await scenario(client)

    return anyio.run(run)


def test_lists_the_three_tools_with_input_schemas(store: InMemoryStore) -> None:
    async def scenario(client: ClientSession) -> Any:
        return (await client.list_tools()).tools

    listed = {tool.name: tool for tool in _with_client(store, None, scenario)}

    assert set(listed) == {"search_documents", "ask_question", "list_documents"}
    assert set(listed["search_documents"].inputSchema["properties"]) == {"query", "k"}
    assert listed["ask_question"].inputSchema["required"] == ["question"]
    assert all(tool.outputSchema for tool in listed.values())


def test_search_documents_returns_structured_hits(store: InMemoryStore) -> None:
    async def scenario(client: ClientSession) -> Any:
        return await client.call_tool(
            "search_documents", {"query": "redevance", "k": 2}
        )

    result = _with_client(store, None, scenario)

    assert not result.isError
    hits = result.structuredContent["hits"]
    assert len(hits) == 2
    assert hits[0]["source_path"] == "/docs/contrat.pdf"


def test_ask_question_returns_answer_and_sources(store: InMemoryStore) -> None:
    async def scenario(client: ClientSession) -> Any:
        return await client.call_tool(
            "ask_question", {"question": "Quel est le montant de la redevance ?"}
        )

    result = _with_client(store, CitingLLM(), scenario)

    assert not result.isError
    content = result.structuredContent
    assert content["answer"] == "La redevance est de 184 000 euros HT."
    assert content["sufficient_context"] is True
    assert [s["chunk_id"] for s in content["sources"]] == [store.chunks[0].chunk_id]


def test_ask_question_without_llm_is_a_tool_error(store: InMemoryStore) -> None:
    async def scenario(client: ClientSession) -> Any:
        return await client.call_tool("ask_question", {"question": "Montant ?"})

    result = _with_client(store, None, scenario)

    assert result.isError
    assert "Aucun fournisseur LLM" in result.content[0].text


def test_invalid_k_is_a_tool_error(store: InMemoryStore) -> None:
    async def scenario(client: ClientSession) -> Any:
        return await client.call_tool("search_documents", {"query": "x", "k": 0})

    result = _with_client(store, None, scenario)

    assert result.isError
    assert "k doit" in result.content[0].text


def test_list_documents(store: InMemoryStore) -> None:
    async def scenario(client: ClientSession) -> Any:
        return await client.call_tool("list_documents", {})

    result = _with_client(store, None, scenario)

    assert [d["chunk_count"] for d in result.structuredContent["documents"]] == [2, 1]
