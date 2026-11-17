"""A LangGraph agent for a data platform: routes questions to SQL, docs (RAG) or a guarded action.

    pip install langgraph

Graph:

    START -> router --sql----> generate_sql -> validate_sql --ok--> run_sql --ok--> answer -> END
                     |                 ^              |                  |
                     |                 +---- retry ---+------ error -----+   (max 3 attempts)
                     +-docs---> retrieve_docs -> answer -> END
                     +-action-> plan_action -> [INTERRUPT: human approval] -> execute_action -> END

Concepts shown: typed state, nodes, conditional edges, a retry loop, checkpointing,
and human-in-the-loop with `interrupt_before`. Every LLM call goes through the
`ailearn.llm.LLM` interface, so tests run offline with `ScriptedLLM`.
"""
from __future__ import annotations

import re
from typing import Callable, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .llm import LLM
from .rag import RAGPipeline

MAX_SQL_ATTEMPTS = 3


class AgentState(TypedDict, total=False):
    question: str
    route: Literal["sql", "docs", "action"]
    sql: str
    sql_error: Optional[str]
    sql_result: str
    attempts: int
    context: str
    action: str
    approved: bool
    answer: str
    trace: list[str]


ROUTER_PROMPT = """Classify the user's request for a data-platform assistant.
- sql: needs numbers/aggregates from warehouse tables
- docs: how-to / conceptual / runbook questions
- action: asks to change something (rerun a pipeline, grant access, delete data)
Reply with exactly one word: sql, docs or action.

Request: {question}"""

SQL_PROMPT = """Write one read-only SQLite SELECT query that answers the question.
<schema>
{schema}
</schema>
{error_hint}Return only the SQL, no explanation.
Question: {question}"""

ANSWER_PROMPT = """Answer the question concisely using only the information below.
<context>
{context}
</context>
<question>{question}</question>"""


def _clean_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip().rstrip(";")


def validate_readonly_sql(sql: str) -> Optional[str]:
    """Return an error message, or None if the SQL is acceptable."""
    if not re.match(r"^\s*(select|with)\b", sql, re.I):
        return "Only SELECT/WITH queries are allowed."
    if ";" in sql:
        return "Only a single statement is allowed."
    if re.search(r"\b(insert|update|delete|drop|alter|create|attach|pragma)\b", sql, re.I):
        return "Write/DDL keywords are not allowed."
    return None


def build_graph(llm: LLM, conn, rag: RAGPipeline | None = None,
                action_executor: Callable[[str], str] | None = None, checkpointer=None):
    """Build and compile the graph. `conn` is a sqlite3 (DB-API) connection."""
    schema = "\n".join(r[0] for r in conn.execute(
        "select sql from sqlite_master where type='table'").fetchall())
    action_executor = action_executor or (lambda a: f"Executed: {a}")

    def log(state: AgentState, msg: str) -> list[str]:
        return [*state.get("trace", []), msg]

    # ---------------- nodes (each returns a partial state update) ----------------
    def router(state: AgentState) -> AgentState:
        out = llm.complete(ROUTER_PROMPT.format(question=state["question"])).strip().lower()
        route = next((r for r in ("sql", "docs", "action") if r in out), "docs")
        return {"route": route, "attempts": 0, "trace": log(state, f"router -> {route}")}

    def generate_sql(state: AgentState) -> AgentState:
        hint = f"The previous query failed with: {state['sql_error']}\nFix it.\n" if state.get("sql_error") else ""
        sql = _clean_sql(llm.complete(SQL_PROMPT.format(schema=schema, question=state["question"], error_hint=hint)))
        return {"sql": sql, "attempts": state.get("attempts", 0) + 1, "trace": log(state, f"generate_sql: {sql}")}

    def validate_sql(state: AgentState) -> AgentState:
        err = validate_readonly_sql(state["sql"])
        return {"sql_error": err, "trace": log(state, f"validate_sql: {err or 'ok'}")}

    def run_sql(state: AgentState) -> AgentState:
        try:
            rows = conn.execute(state["sql"]).fetchmany(50)
            return {"sql_result": str(rows), "sql_error": None, "context": f"SQL: {state['sql']}\nRows: {rows}",
                    "trace": log(state, f"run_sql: {rows}")}
        except Exception as e:  # feed DB errors back to the LLM
            return {"sql_error": f"{type(e).__name__}: {e}", "trace": log(state, f"run_sql error: {e}")}

    def retrieve_docs(state: AgentState) -> AgentState:
        results = rag.retrieve(state["question"]) if rag else []
        ctx = "\n".join(f"[{i}] {r.text}" for i, r in enumerate(results, 1)) or "(no documents found)"
        return {"context": ctx, "trace": log(state, f"retrieve_docs: {len(results)} chunks")}

    def answer(state: AgentState) -> AgentState:
        if state.get("sql_error") and state.get("route") == "sql":
            return {"answer": f"Sorry, I couldn't build a valid query after {state['attempts']} attempts "
                              f"(last error: {state['sql_error']}).", "trace": log(state, "answer: gave up")}
        text = llm.complete(ANSWER_PROMPT.format(context=state.get("context", ""), question=state["question"]))
        return {"answer": text.strip(), "trace": log(state, "answer")}

    def plan_action(state: AgentState) -> AgentState:
        action = llm.complete(f"Describe, in one line, the exact operation to perform for: {state['question']}").strip()
        return {"action": action, "approved": False, "trace": log(state, f"plan_action: {action}")}

    def execute_action(state: AgentState) -> AgentState:
        if not state.get("approved"):
            return {"answer": f"Action NOT executed (rejected): {state['action']}",
                    "trace": log(state, "execute_action: rejected")}
        result = action_executor(state["action"])
        return {"answer": result, "trace": log(state, "execute_action: done")}

    # ---------------- conditional edges ----------------
    def after_validate(state: AgentState) -> str:
        if state.get("sql_error") is None:
            return "run_sql"
        return "generate_sql" if state["attempts"] < MAX_SQL_ATTEMPTS else "answer"

    def after_run(state: AgentState) -> str:
        if state.get("sql_error") is None:
            return "answer"
        return "generate_sql" if state["attempts"] < MAX_SQL_ATTEMPTS else "answer"

    g = StateGraph(AgentState)
    for name, fn in [("router", router), ("generate_sql", generate_sql), ("validate_sql", validate_sql),
                     ("run_sql", run_sql), ("retrieve_docs", retrieve_docs), ("answer", answer),
                     ("plan_action", plan_action), ("execute_action", execute_action)]:
        g.add_node(name, fn)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: s["route"],
                            {"sql": "generate_sql", "docs": "retrieve_docs", "action": "plan_action"})
    g.add_edge("generate_sql", "validate_sql")
    g.add_conditional_edges("validate_sql", after_validate, ["run_sql", "generate_sql", "answer"])
    g.add_conditional_edges("run_sql", after_run, ["answer", "generate_sql"])
    g.add_edge("retrieve_docs", "answer")
    g.add_edge("answer", END)
    g.add_edge("plan_action", "execute_action")
    g.add_edge("execute_action", END)

    # Pause BEFORE any side effect so a human can approve/reject (needs a checkpointer).
    return g.compile(checkpointer=checkpointer or MemorySaver(), interrupt_before=["execute_action"])
