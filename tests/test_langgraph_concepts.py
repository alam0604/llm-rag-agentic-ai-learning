"""LEARNING TESTS for LangGraph: each test demonstrates one concept from
docs/deep-dives/01-langgraph-in-depth.md. Read them like executable notes."""
import operator
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, Send, interrupt  # noqa: E402


def cfg(t="t"):
    return {"configurable": {"thread_id": t}}


# 1. Reducers: by default a key is OVERWRITTEN; with Annotated[..., reducer] updates are MERGED.
class LogState(TypedDict):
    last: str
    log: Annotated[list[str], operator.add]


def test_reducers_overwrite_vs_append():
    g = StateGraph(LogState)
    g.add_node("a", lambda s: {"last": "a", "log": ["a"]})
    g.add_node("b", lambda s: {"last": "b", "log": ["b"]})
    g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", END)
    out = g.compile().invoke({"last": "", "log": ["start"]})
    assert out["last"] == "b"                         # overwritten
    assert out["log"] == ["start", "a", "b"]          # appended by operator.add


# 2. Parallel fan-out / map-reduce with Send: one worker per item, results merged by the reducer.
class MapState(TypedDict):
    tables: list[str]
    results: Annotated[list[str], operator.add]


def test_send_map_reduce_in_parallel():
    def fan_out(s):
        return [Send("profile", {"table": t}) for t in s["tables"]]

    g = StateGraph(MapState)
    g.add_node("profile", lambda s: {"results": [f"{s['table']}:ok"]})
    g.add_conditional_edges(START, fan_out, ["profile"])
    g.add_edge("profile", END)
    out = g.compile().invoke({"tables": ["bronze", "silver", "gold"], "results": []})
    assert sorted(out["results"]) == ["bronze:ok", "gold:ok", "silver:ok"]


# 3. interrupt() inside a node + Command(resume=...) — the modern human-in-the-loop API.
class ApprovalState(TypedDict):
    action: str
    status: str


def test_interrupt_and_resume_with_command():
    def approve(s):
        decision = interrupt({"question": f"Approve '{s['action']}'?"})   # pauses the graph here
        return {"status": "done" if decision == "yes" else "rejected"}

    g = StateGraph(ApprovalState)
    g.add_node("approve", approve)
    g.add_edge(START, "approve"); g.add_edge("approve", END)
    app = g.compile(checkpointer=MemorySaver())

    first = app.invoke({"action": "rerun pipeline", "status": "new"}, cfg("h1"))
    assert first["__interrupt__"][0].value == {"question": "Approve 'rerun pipeline'?"}
    out = app.invoke(Command(resume="yes"), cfg("h1"))                    # human answers
    assert out["status"] == "done"


# 4. Command from a node: update state AND choose the next node in one return (dynamic routing).
class RouteState(TypedDict):
    q: str
    path: Annotated[list[str], operator.add]


def test_command_goto_routing():
    def router(s) -> Command:
        return Command(update={"path": ["router"]}, goto="sql" if "total" in s["q"] else "docs")

    g = StateGraph(RouteState)
    g.add_node("router", router, destinations=("sql", "docs"))
    g.add_node("sql", lambda s: {"path": ["sql"]})
    g.add_node("docs", lambda s: {"path": ["docs"]})
    g.add_edge(START, "router"); g.add_edge("sql", END); g.add_edge("docs", END)
    app = g.compile()
    assert app.invoke({"q": "total revenue", "path": []})["path"] == ["router", "sql"]
    assert app.invoke({"q": "how to vacuum", "path": []})["path"] == ["router", "docs"]


# 5. Streaming modes: "updates" = per-node deltas, "values" = full state after each step.
def test_stream_modes():
    g = StateGraph(LogState)
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", END)
    app = g.compile()
    updates = list(app.stream({"last": "", "log": []}, stream_mode="updates"))
    assert updates == [{"a": {"log": ["a"]}}, {"b": {"log": ["b"]}}]
    values = list(app.stream({"last": "", "log": []}, stream_mode="values"))
    assert values[-1]["log"] == ["a", "b"]


# 6. Checkpoints = memory across turns + history for time travel / debugging.
class ChatState(TypedDict):
    turns: Annotated[list[str], operator.add]


def test_checkpointer_gives_memory_and_history():
    g = StateGraph(ChatState)
    g.add_node("bot", lambda s: {"turns": [f"bot saw {len(s['turns'])} turns"]})
    g.add_edge(START, "bot"); g.add_edge("bot", END)
    app = g.compile(checkpointer=MemorySaver())
    app.invoke({"turns": ["user: hi"]}, cfg("c1"))
    out = app.invoke({"turns": ["user: again"]}, cfg("c1"))     # same thread -> remembers
    assert out["turns"][-1] == "bot saw 3 turns"
    assert app.invoke({"turns": ["user: hi"]}, cfg("c2"))["turns"][-1] == "bot saw 1 turns"  # isolated thread
    history = list(app.get_state_history(cfg("c1")))
    assert len(history) >= 4                                    # every step was checkpointed


# 7. Subgraphs: a compiled graph can be a node of another graph (team of agents).
def test_subgraph_as_node():
    sub = StateGraph(LogState)
    sub.add_node("inner", lambda s: {"log": ["inner"]})
    sub.add_edge(START, "inner"); sub.add_edge("inner", END)

    sub_app = sub.compile()

    def build(team_node):
        parent = StateGraph(LogState)
        parent.add_node("before", lambda s: {"log": ["before"]})
        parent.add_node("team", team_node)
        parent.add_edge(START, "before"); parent.add_edge("before", "team"); parent.add_edge("team", END)
        return parent.compile()

    # GOTCHA: added directly, the subgraph returns its WHOLE `log` (["before", "inner"]) and the
    # parent's operator.add reducer appends it again -> "before" is duplicated.
    assert build(sub_app).invoke({"last": "", "log": []})["log"] == ["before", "before", "inner"]

    # FIX: call the subgraph inside a node and return only what is new.
    def team(s):
        out = sub_app.invoke({"last": "", "log": []})
        return {"log": out["log"]}
    assert build(team).invoke({"last": "", "log": []})["log"] == ["before", "inner"]
