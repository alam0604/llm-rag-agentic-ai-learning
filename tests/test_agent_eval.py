import sqlite3

import pytest

from ailearn.agent import ReActAgent, Tool, make_lookup_tool, make_sql_tool, safe_calculator
from ailearn.evaluation import (evaluate_retriever, groundedness, hit_rate_at_k, mrr, ndcg_at_k,
                                precision_at_k, recall_at_k, token_f1)
from ailearn.llm import ScriptedLLM
from ailearn.rag import RAGPipeline


# ---------------- agent ----------------
def calc_tool():
    return Tool("calculator", "Evaluate arithmetic", safe_calculator)


def test_agent_uses_tool_then_answers():
    llm = ScriptedLLM([
        "Thought: I need to multiply.\nAction: calculator[23 * 19]",
        "Thought: I now know the answer\nFinal Answer: 437",
    ])
    res = ReActAgent(llm, [calc_tool()]).run("What is 23*19?")
    assert res.answer == "437"
    assert res.steps[0].action == "calculator" and res.steps[0].observation == "437"
    assert "Observation: 437" in llm.calls[1]  # observation is fed back to the LLM


def test_agent_multi_tool_chain():
    lookup = make_lookup_tool({"bronze_orders": "1200"}, "row_count", "Row count of a table")
    llm = ScriptedLLM([
        "Thought: get rows\nAction: row_count[bronze_orders]",
        "Thought: 3 days\nAction: calculator[1200 * 3]",
        "Final Answer: 3600 rows",
    ])
    res = ReActAgent(llm, [lookup, calc_tool()]).run("rows in 3 days?")
    assert res.answer == "3600 rows" and [s.action for s in res.steps[:2]] == ["row_count", "calculator"]


def test_agent_recovers_from_unknown_tool_and_tool_errors():
    llm = ScriptedLLM([
        "Thought: x\nAction: web_search[weather]",
        "Thought: x\nAction: calculator[import os]",
        "Final Answer: done",
    ])
    res = ReActAgent(llm, [calc_tool()]).run("q")
    assert "unknown tool" in res.steps[0].observation
    assert res.steps[1].observation.startswith("Error:")
    assert res.answer == "done"


def test_agent_stops_at_max_steps():
    llm = ScriptedLLM(["Thought: loop\nAction: calculator[1+1]"] * 3)
    res = ReActAgent(llm, [calc_tool()], max_steps=3).run("q")
    assert res.answer is None and res.stopped_reason == "max_steps"


def test_agent_handles_unparseable_output():
    llm = ScriptedLLM(["I think the answer is maybe 4", "Final Answer: 4"])
    res = ReActAgent(llm, [calc_tool()]).run("2+2")
    assert "could not parse" in res.steps[0].observation and res.answer == "4"


@pytest.mark.parametrize("expr,expected", [("2+3*4", "14"), ("(2+3)*4", "20"), ("10/4", "2.5"), ("-3**2", "-9")])
def test_safe_calculator(expr, expected):
    assert safe_calculator(expr) == expected


@pytest.mark.parametrize("bad", ["__import__('os')", "open('x')", "a+1"])
def test_safe_calculator_rejects_code(bad):
    with pytest.raises(Exception):
        safe_calculator(bad)


def test_sql_tool_is_read_only():
    conn = sqlite3.connect(":memory:")
    conn.execute("create table sales(region text, amount int)")
    conn.executemany("insert into sales values (?,?)", [("north", 10), ("south", 30)])
    tool = make_sql_tool(conn)
    assert tool.func("select sum(amount) from sales") == "[(40,)]"
    with pytest.raises(PermissionError):
        tool.func("drop table sales")
    with pytest.raises(PermissionError):
        tool.func("select 1; drop table sales")


# ---------------- evaluation ----------------
def test_retrieval_metrics():
    got, rel = ["a", "x", "b", "y"], {"a", "b", "c"}
    assert precision_at_k(got, rel, 2) == 0.5
    assert recall_at_k(got, rel, 4) == pytest.approx(2 / 3)
    assert hit_rate_at_k(["x", "y"], rel, 2) == 0.0
    assert mrr(["x", "b"], rel) == 0.5
    assert ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], {"a": 3, "b": 1}, 2) < 1.0


def test_token_f1_and_groundedness():
    assert token_f1("gold layer", "the gold layer") == 1.0
    assert token_f1("bronze", "gold") == 0.0
    ctx = ["Gold holds business-level aggregates ready for reporting."]
    assert groundedness("Gold holds business-level aggregates. [1]", ctx) == 1.0
    assert groundedness("Snowflake invented gold layers in 1999.", ctx) < 1.0


def test_evaluate_retriever_on_golden_set(docs):
    rag = RAGPipeline(top_k=3)
    rag.add_documents(docs)
    golden = [
        {"question": "time travel older versions", "relevant": {"delta#0"}},
        {"question": "bronze silver gold", "relevant": {"medallion#0"}},
        {"question": "pipelines activities triggers", "relevant": {"adf#0"}},
    ]
    report = evaluate_retriever(lambda q, k: [r.id for r in rag.retrieve(q, k)], golden, k=3)
    assert report["hit_rate@k"] == 1.0 and report["mrr"] > 0.5
