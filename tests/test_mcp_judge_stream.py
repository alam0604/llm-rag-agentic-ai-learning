import asyncio
import json
import sqlite3

import pytest

from ailearn.llm import ScriptedLLM
from ailearn.llm_judge import llm_judge, run_eval
from ailearn.rag import RAGPipeline


# ---------------- MCP server (in-memory client, no subprocess) ----------------
mcp = pytest.importorskip("mcp")
try:
    from mcp.client.client import Client
    from ailearn.mcp_server import build_server
except ImportError:  # pragma: no cover - SDK 1.x installed
    pytest.skip("needs MCP Python SDK 2.x", allow_module_level=True)


@pytest.fixture
def server(docs):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript("""create table gold_daily_sales(sale_date text, region text, revenue real);
                          insert into gold_daily_sales values ('d1','north',10),('d1','south',5);""")
    rag = RAGPipeline(top_k=2)
    rag.add_documents(docs)
    return build_server(conn, rag)


def run(coro):
    return asyncio.run(coro)


def test_mcp_lists_tools_resources_prompts(server):
    async def go():
        async with Client(server) as c:
            tools = {t.name for t in (await c.list_tools()).tools}
            res = [str(r.uri) for r in (await c.list_resources()).resources]
            prompts = [p.name for p in (await c.list_prompts()).prompts]
            return tools, res, prompts
    tools, res, prompts = run(go())
    assert tools == {"list_tables", "describe_table", "run_readonly_query", "search_docs"}
    assert res == ["warehouse://schema"] and prompts == ["analyze_table"]


def test_mcp_tools_work_end_to_end(server):
    async def go():
        async with Client(server) as c:
            q = await c.call_tool("run_readonly_query", {"sql": "select region, revenue from gold_daily_sales order by revenue desc"})
            d = await c.call_tool("describe_table", {"table": "gold_daily_sales"})
            s = await c.call_tool("search_docs", {"query": "bronze silver gold layers", "k": 1})
            schema = await c.read_resource("warehouse://schema")
            return q, d, s, schema
    q, d, s, schema = run(go())
    assert q.structured_content["result"][0] == {"region": "north", "revenue": 10.0}
    assert {"column": "revenue", "type": "REAL"} in d.structured_content["result"]
    assert s.structured_content["result"][0]["id"].startswith("medallion")
    assert "CREATE TABLE gold_daily_sales" in schema.contents[0].text


@pytest.mark.parametrize("args", [
    ("run_readonly_query", {"sql": "drop table gold_daily_sales"}),
    ("run_readonly_query", {"sql": "select 1; delete from gold_daily_sales"}),
    ("describe_table", {"table": "x; drop table y"}),
])
def test_mcp_guardrails_return_tool_errors(server, args):
    async def go():
        async with Client(server) as c:
            return await c.call_tool(*args)
    result = run(go())
    assert result.is_error  # surfaced to the model as an error result, not a crash


# ---------------- LLM-as-judge & eval runner ----------------
def test_llm_judge_parses_verdict():
    judge = ScriptedLLM(['{"score": 2, "reasoning": "Claims Snowflake invented it; context does not say so."}'])
    v = llm_judge(judge, "faithfulness", "who invented gold layers?", "Snowflake did.", context="Gold holds aggregates.")
    assert v.score == 2 and "Snowflake" in v.reasoning
    assert "Rubric" in judge.calls[0] and "CONTEXT: Gold holds aggregates." in judge.calls[0]


def test_run_eval_aggregates_and_gates_regressions():
    app = lambda q: ("Gold holds business-level aggregates. [1]", "Gold holds business-level aggregates.")
    judge = ScriptedLLM(['{"score": 5, "reasoning": "ok"}', '{"score": 4, "reasoning": "ok"}'] * 2)
    data = [{"question": "What is in gold?", "reference": "business-level aggregates"},
            {"question": "Gold layer contents?"}]
    rep = run_eval(app, data, judge=judge, thresholds={"faithfulness": 0.9},
                   baseline={"answer_relevance": 0.95})
    assert rep.metrics["faithfulness"] == 1.0
    assert rep.metrics["answer_relevance"] == 0.75
    assert rep.metrics["groundedness_lexical"] == 1.0
    assert not rep.passed and "answer_relevance dropped" in rep.failures[0]
    assert "| faithfulness | 1.0 |" in rep.to_markdown()
    assert json.loads(rep.to_json())["metrics"]["faithfulness"] == 1.0


# ---------------- SSE streaming endpoint ----------------
def test_ask_stream_emits_node_events():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ailearn.api import create_app
    client = TestClient(create_app())
    with client.stream("POST", "/ask/stream", json={"question": "total revenue by region"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    nodes = [json.loads(line[6:])["node"] for line in body.splitlines()
             if line.startswith("data: ") and '"node"' in line]
    assert nodes == ["router", "generate_sql", "validate_sql", "run_sql", "answer"]
    assert "event: result" in body and '"status":"done"' in body
