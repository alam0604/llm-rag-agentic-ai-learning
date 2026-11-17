"""Serve the LangGraph agent as a REST API with human-in-the-loop approvals.

    pip install fastapi uvicorn langgraph
    uvicorn ailearn.api:app --reload          # offline demo (scripted LLM)
    USE_REAL_LLM=1 OPENAI_API_KEY=... uvicorn ailearn.api:app

Endpoints
  POST /ask                       {"question": "...", "thread_id": "optional"}
  POST /approve/{thread_id}       {"approved": true|false}
  GET  /threads/{thread_id}       current state (debug)
  GET  /health
"""
from __future__ import annotations

import os
import sqlite3
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .graph_agent import build_graph
from .llm import LLM


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    thread_id: str | None = None


class ApproveRequest(BaseModel):
    approved: bool


class AgentResponse(BaseModel):
    thread_id: str
    status: str                      # "done" | "pending_approval"
    answer: str | None = None
    pending_action: str | None = None
    trace: list[str] = []


class _DemoLLM:
    """Tiny rule-based stand-in so the API runs without any key."""

    def complete(self, prompt: str) -> str:
        if prompt.startswith("Classify"):
            q = prompt.lower().split("request:")[-1]
            if any(w in q for w in ("rerun", "delete", "grant", "restart")):
                return "action"
            return "sql" if any(w in q for w in ("how many", "total", "revenue", "count", "sum")) else "docs"
        if prompt.startswith("Write one read-only"):
            return "select region, sum(revenue) as revenue from gold_daily_sales group by region order by 2 desc"
        if prompt.startswith("Describe, in one line"):
            return "Rerun pipeline: " + prompt.split("for:")[-1].strip()
        return "Based on the data: " + prompt.split("<context>")[-1].split("</context>")[0].strip()[:300]


def _demo_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript("""
        create table gold_daily_sales(sale_date text, region text, revenue real);
        insert into gold_daily_sales values ('2026-11-01','north',12000),('2026-11-01','south',8000),
                                            ('2026-11-02','north',13500),('2026-11-02','south',9100);
    """)
    return conn


def create_app(llm: LLM | None = None, conn=None, rag=None, action_executor=None) -> FastAPI:
    if llm is None and os.getenv("USE_REAL_LLM"):
        from .llm import OpenAICompatibleLLM
        llm = OpenAICompatibleLLM(model=os.getenv("MODEL", "gpt-4o-mini"), base_url=os.getenv("OPENAI_BASE_URL"))
    graph = build_graph(llm or _DemoLLM(), conn or _demo_db(), rag=rag, action_executor=action_executor)
    app = FastAPI(title="Data Platform Agent", version="1.0")

    def _respond(thread_id: str) -> AgentResponse:
        snap = graph.get_state({"configurable": {"thread_id": thread_id}})
        s = snap.values
        if snap.next == ("execute_action",):
            return AgentResponse(thread_id=thread_id, status="pending_approval",
                                 pending_action=s.get("action"), trace=s.get("trace", []))
        return AgentResponse(thread_id=thread_id, status="done", answer=s.get("answer"), trace=s.get("trace", []))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ask", response_model=AgentResponse)
    def ask(req: AskRequest):
        thread_id = req.thread_id or str(uuid.uuid4())
        graph.invoke({"question": req.question, "trace": []}, {"configurable": {"thread_id": thread_id}})
        return _respond(thread_id)

    @app.post("/approve/{thread_id}", response_model=AgentResponse)
    def approve(thread_id: str, req: ApproveRequest):
        config = {"configurable": {"thread_id": thread_id}}
        if graph.get_state(config).next != ("execute_action",):
            raise HTTPException(409, "No action is waiting for approval on this thread")
        graph.update_state(config, {"approved": req.approved})
        graph.invoke(None, config)
        return _respond(thread_id)

    @app.get("/threads/{thread_id}", response_model=AgentResponse)
    def thread(thread_id: str):
        if not graph.get_state({"configurable": {"thread_id": thread_id}}).values:
            raise HTTPException(404, "Unknown thread")
        return _respond(thread_id)

    return app


try:  # module-level app for `uvicorn ailearn.api:app`
    app = create_app()
except ImportError:  # pragma: no cover
    app = None
