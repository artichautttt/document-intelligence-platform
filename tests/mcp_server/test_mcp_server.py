"""Outils appelés par un vrai client MCP (`ClientSession`), connecté en mémoire au serveur."""

from collections.abc import Awaitable, Callable
from typing import Any

import anyio
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

from document_intelligence.mcp_server import build_server
from document_intelligence.orchestration.backends import LangGraphBackend
from document_intelligence.orchestration.llm import LLMClient

from .conftest import CitingLLM, InMemoryStore


def _with_client(
    store: InMemoryStore,
    llm: LLMClient | None,
    scenario: Callable[[ClientSession], Awaitable[Any]],
) -> Any:
    backend = LangGraphBackend(store, llm) if llm is not None else None

    async def run() -> Any:
        async with create_connected_server_and_client_session(
            build_server(store, backend)
        ) as client:
            return await scenario(client)

    return anyio.run(run)


def test_lists_the_tools_with_input_schemas(store: InMemoryStore) -> None:
    async def scenario(client: ClientSession) -> Any:
        return (await client.list_tools()).tools

    listed = {tool.name: tool for tool in _with_client(store, None, scenario)}

    assert set(listed) == {
        "search_documents",
        "ask_question",
        "list_documents",
        "describe_tables",
        "run_sql",
        "ask_data",
    }
    assert listed["run_sql"].inputSchema["required"] == ["sql"]
    assert listed["ask_data"].inputSchema["required"] == ["question"]
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


def test_describe_tables(store: InMemoryStore) -> None:
    async def scenario(client: ClientSession) -> Any:
        return await client.call_tool("describe_tables", {})

    result = _with_client(store, None, scenario)

    assert not result.isError
    names = [t["name"] for t in result.structuredContent["tables"]]
    assert "od_monthly" in names and "build_info" not in names


def test_run_sql_executes_a_select(store: InMemoryStore) -> None:
    async def scenario(client: ClientSession) -> Any:
        return await client.call_tool(
            "run_sql",
            {"sql": "SELECT COUNT(*) AS n FROM zones WHERE borough = 'Brooklyn'"},
        )

    result = _with_client(store, None, scenario)

    assert not result.isError
    assert result.structuredContent["rows"] == [[61]]
    assert result.structuredContent["truncated"] is False


def test_run_sql_write_is_an_explicit_tool_error(store: InMemoryStore) -> None:
    async def scenario(client: ClientSession) -> Any:
        return await client.call_tool("run_sql", {"sql": "DROP TABLE zones"})

    result = _with_client(store, None, scenario)

    assert result.isError
    assert "lecture seule" in result.content[0].text


def test_ask_data_without_agent_is_a_tool_error(store: InMemoryStore) -> None:
    async def scenario(client: ClientSession) -> Any:
        return await client.call_tool("ask_data", {"question": "Combien de zones ?"})

    result = _with_client(store, None, scenario)

    assert result.isError
    assert "ask_data est indisponible" in result.content[0].text


def test_ask_data_with_a_scripted_agent(store: InMemoryStore) -> None:
    from document_intelligence.data_agent.sql_guard import SafeSQLExecutor
    from document_intelligence.evaluation.scripted_sql_client import ScriptedSQLClient
    from document_intelligence.evaluation.sql_eval_cases import load_sql_eval
    from document_intelligence.orchestration.data_agent import DataAgent

    cases = load_sql_eval()
    brooklyn = next(c for c in cases if c.id == "s04")
    executor = SafeSQLExecutor()
    agent = DataAgent(ScriptedSQLClient(cases), "scripted", executor)

    async def run() -> Any:
        async with create_connected_server_and_client_session(
            build_server(store, None, sql_executor=executor, data_agent=agent)
        ) as client:
            return await client.call_tool("ask_data", {"question": brooklyn.question})

    result = anyio.run(run)

    assert not result.isError
    content = result.structuredContent
    assert content["value"] == 61 and content["refused"] is False
    assert content["sql"] == brooklyn.reference_sql
    assert brooklyn.reference_sql in content["answer"]
    assert content["blocked_queries"] == 0
