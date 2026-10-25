"""Agent text-to-SQL (tool calling natif) avec un client Anthropic scripté."""

import copy
import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from document_intelligence.data_agent import DEFAULT_DB_PATH
from document_intelligence.data_agent.sql_guard import SafeSQLExecutor
from document_intelligence.orchestration.cost import CostLedger, cost_usd
from document_intelligence.orchestration.data_agent import (
    DATA_AGENT_ROLE,
    TOOL_DEFINITIONS,
    DataAgent,
)
from document_intelligence.orchestration.exceptions import LLMGenerationError

MODEL = "claude-sonnet-5"
BROOKLYN = "SELECT COUNT(*) FROM zones WHERE borough = 'Brooklyn'"


def tool_use(block_id: str, name: str, **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=arguments)


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def response(*blocks: SimpleNamespace, stop: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=50),
    )


def submit(**arguments: Any) -> SimpleNamespace:
    return response(tool_use("s", "submit_answer", **arguments))


class ScriptedClient:
    def __init__(self, *responses: SimpleNamespace | Exception) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(copy.deepcopy(kwargs))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(scope="module")
def executor() -> SafeSQLExecutor:
    return SafeSQLExecutor(DEFAULT_DB_PATH)


def _last_tool_result(client: ScriptedClient, index: int) -> dict[str, Any]:
    result: dict[str, Any] = client.requests[index]["messages"][-1]["content"][0]
    return result


def test_executes_sql_and_cites_the_executed_query(executor: SafeSQLExecutor) -> None:
    client = ScriptedClient(
        response(text("Je compte."), tool_use("t1", "run_sql", sql=BROOKLYN)),
        submit(
            answer="Brooklyn compte 61 zones.", value=61, sql=BROOKLYN, refused=False
        ),
    )
    ledger = CostLedger()
    result = DataAgent(client, MODEL, executor, ledger=ledger).run(
        "Zones de Brooklyn ?"
    )

    payload = json.loads(_last_tool_result(client, 1)["content"])
    assert payload["rows"] == [[61]]
    assert result.value == 61 and result.refused is False and result.submitted
    assert result.cited_sql == BROOKLYN
    assert "```sql\n" + BROOKLYN + "\n```" in result.formatted()
    assert result.steps == 2 and result.stop_reason == "submitted"
    assert result.input_tokens == 2000 and result.output_tokens == 100
    assert result.cost_usd == round(cost_usd(MODEL, 2000, 100), 6)
    assert [r.role for r in ledger.records] == [DATA_AGENT_ROLE, DATA_AGENT_ROLE]


def test_request_forces_a_tool_and_embeds_the_schema(executor: SafeSQLExecutor) -> None:
    client = ScriptedClient(submit(answer="x", value=None, refused=True))
    DataAgent(client, MODEL, executor).run("Question ?")
    request = client.requests[0]
    assert request["tool_choice"] == {"type": "any"}
    assert request["tools"] == TOOL_DEFINITIONS
    assert "Table od_monthly" in request["system"]
    assert "sqlite_master" not in request["system"]


def test_blocked_query_is_returned_to_the_model_which_can_retry(
    executor: SafeSQLExecutor,
) -> None:
    client = ScriptedClient(
        response(tool_use("t1", "run_sql", sql="DELETE FROM zones")),
        response(tool_use("t2", "run_sql", sql="SELECT name FROM sqlite_master")),
        response(tool_use("t3", "run_sql", sql=BROOKLYN)),
        submit(answer="61 zones.", value=61, sql=BROOKLYN, refused=False),
    )
    result = DataAgent(client, MODEL, executor).run("Zones de Brooklyn ?")

    first, second = _last_tool_result(client, 1), _last_tool_result(client, 2)
    assert first["is_error"] and "write_or_admin_keyword" in first["content"]
    assert second["is_error"] and "table_not_allowed" in second["content"]
    assert [c.rule for c in result.blocked_calls] == [
        "write_or_admin_keyword",
        "table_not_allowed",
    ]
    assert result.value == 61 and result.cited_sql == BROOKLYN


def test_sql_error_is_not_counted_as_blocked(executor: SafeSQLExecutor) -> None:
    client = ScriptedClient(
        response(tool_use("t1", "run_sql", sql="SELECT nope FROM zones")),
        submit(answer="Échec.", value=None, refused=True),
    )
    result = DataAgent(client, MODEL, executor).run("?")
    assert _last_tool_result(client, 1)["is_error"]
    assert result.blocked_calls == []
    assert result.sql_calls[0].ok is False and "no such column" in (
        result.sql_calls[0].error or ""
    )


def test_cannot_cite_a_query_it_did_not_execute(executor: SafeSQLExecutor) -> None:
    invented = "SELECT 999 FROM zones"
    client = ScriptedClient(
        response(tool_use("t1", "run_sql", sql=BROOKLYN)),
        submit(answer="61.", value=61, sql=invented, refused=False),
    )
    result = DataAgent(client, MODEL, executor).run("?")
    assert result.cited_sql == BROOKLYN


def test_refusal_has_no_value_and_no_cited_query(executor: SafeSQLExecutor) -> None:
    client = ScriptedClient(
        response(tool_use("t1", "run_sql", sql=BROOKLYN)),
        submit(answer="Pas de données météo.", value=12.5, sql=BROOKLYN, refused=True),
    )
    result = DataAgent(client, MODEL, executor).run("Température ?")
    assert result.refused and result.value is None and result.cited_sql is None
    assert result.formatted() == "Pas de données météo."


def test_end_turn_without_submission(executor: SafeSQLExecutor) -> None:
    client = ScriptedClient(response(text("Je ne sais pas."), stop="end_turn"))
    result = DataAgent(client, MODEL, executor).run("?")
    assert result.submitted is False and result.stop_reason == "end_turn"
    assert result.value is None


def test_max_steps_bounds_the_loop(executor: SafeSQLExecutor) -> None:
    client = ScriptedClient(
        *(response(tool_use(f"t{i}", "run_sql", sql=BROOKLYN)) for i in range(3))
    )
    result = DataAgent(client, MODEL, executor, max_steps=3).run("?")
    assert result.stop_reason == "max_steps" and result.steps == 3
    assert len(result.sql_calls) == 3 and result.submitted is False


def test_unknown_tool_is_an_error_for_the_model(executor: SafeSQLExecutor) -> None:
    client = ScriptedClient(
        response(tool_use("t1", "drop_everything")),
        submit(answer="ok", value=None, refused=True),
    )
    DataAgent(client, MODEL, executor).run("?")
    assert _last_tool_result(client, 1)["is_error"]


def test_api_error_is_raised(executor: SafeSQLExecutor) -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client = ScriptedClient(anthropic.APIConnectionError(request=request))
    with pytest.raises(LLMGenerationError):
        DataAgent(client, MODEL, executor).run("?")
