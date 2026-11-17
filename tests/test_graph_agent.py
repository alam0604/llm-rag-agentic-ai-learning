import sqlite3

import pytest

pytest.importorskip("langgraph")

from ailearn.graph_agent import build_graph, validate_readonly_sql  # noqa: E402
from ailearn.llm import ScriptedLLM  # noqa: E402
from ailearn.rag import RAGPipeline  # noqa: E402


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.executescript("""
        create table gold_daily_sales(sale_date text, region text, revenue real);
        insert into gold_daily_sales values ('2026-11-01','north',100),('2026-11-01','south',40),
                                            ('2026-11-02','north',50),('2026-11-02','south',60);
    """)
    return c


def cfg(thread="t1"):
    return {"configurable": {"thread_id": thread}}


def test_sql_route_happy_path(conn):
    llm = ScriptedLLM(["sql",
                       "```sql\nselect region, sum(revenue) from gold_daily_sales group by region order by 2 desc\n```",
                       "North has the highest revenue (150)."])
    out = build_graph(llm, conn).invoke({"question": "Which region has the most revenue?"}, cfg())
    assert out["route"] == "sql"
    assert out["sql_result"] == "[('north', 150.0), ('south', 100.0)]"
    assert out["answer"].startswith("North")
    assert out["attempts"] == 1


def test_sql_self_corrects_after_db_error(conn):
    llm = ScriptedLLM(["sql",
                       "select sum(amount) from gold_daily_sales",          # wrong column
                       "select sum(revenue) from gold_daily_sales",         # fixed
                       "Total revenue is 250."])
    out = build_graph(llm, conn).invoke({"question": "total revenue?"}, cfg())
    assert out["attempts"] == 2 and out["sql_result"] == "[(250.0,)]"
    assert "previous query failed" in llm.calls[2].lower()   # the error was fed back to the LLM


def test_destructive_sql_is_blocked_and_gives_up(conn):
    llm = ScriptedLLM(["sql", "drop table gold_daily_sales", "delete from gold_daily_sales",
                       "select 1; drop table gold_daily_sales"])
    out = build_graph(llm, conn).invoke({"question": "clean up the table"}, cfg())
    assert out["attempts"] == 3 and "couldn't build a valid query" in out["answer"]
    assert conn.execute("select count(*) from gold_daily_sales").fetchone()[0] == 4   # table intact


def test_docs_route_uses_rag(conn, docs):
    rag = RAGPipeline(top_k=1)
    rag.add_documents(docs)
    llm = ScriptedLLM(["docs", "Gold holds business-level aggregates."])
    out = build_graph(llm, conn, rag=rag).invoke({"question": "What does the gold layer hold?"}, cfg())
    assert out["route"] == "docs" and "Gold" in out["context"]
    assert "<context>" in llm.calls[1] and "Gold" in llm.calls[1]


def test_action_pauses_for_human_approval_then_executes(conn):
    executed = []
    llm = ScriptedLLM(["action", "Rerun pipeline pl_ingest_sales for 2026-11-17"])
    graph = build_graph(llm, conn, action_executor=lambda a: executed.append(a) or f"Executed: {a}")
    graph.invoke({"question": "rerun yesterday's sales ingestion"}, cfg("a1"))

    snap = graph.get_state(cfg("a1"))
    assert snap.next == ("execute_action",) and executed == []        # paused, nothing done yet

    graph.update_state(cfg("a1"), {"approved": True})                  # human approves
    out = graph.invoke(None, cfg("a1"))                                # resume from checkpoint
    assert executed == ["Rerun pipeline pl_ingest_sales for 2026-11-17"]
    assert out["answer"].startswith("Executed")


def test_action_rejected_by_human(conn):
    executed = []
    llm = ScriptedLLM(["action", "Drop table gold_daily_sales"])
    graph = build_graph(llm, conn, action_executor=lambda a: executed.append(a))
    graph.invoke({"question": "delete the sales table"}, cfg("a2"))
    out = graph.invoke(None, cfg("a2"))                                # resumed without approval
    assert executed == [] and "NOT executed" in out["answer"]


def test_router_falls_back_to_docs_on_garbage(conn):
    llm = ScriptedLLM(["¯\\_(ツ)_/¯", "I don't know."])
    out = build_graph(llm, conn).invoke({"question": "hmm"}, cfg())
    assert out["route"] == "docs"


@pytest.mark.parametrize("sql,ok", [
    ("select * from t", True), ("with x as (select 1) select * from x", True),
    ("update t set a=1", False), ("select 1; drop table t", False), ("select * from t where 1=1 or drop", False),
])
def test_validate_readonly_sql(sql, ok):
    assert (validate_readonly_sql(sql) is None) == ok
