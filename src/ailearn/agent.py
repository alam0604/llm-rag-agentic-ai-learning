"""A ReAct agent (Reason + Act) from scratch.

Loop:
    Thought: what should I do next?
    Action: tool_name[input]          <- the LLM *chooses* a tool
    Observation: <tool output>        <- we run the tool and feed the result back
    ... repeat ...
    Final Answer: <answer>

Modern APIs expose this as native "tool calling / function calling" with JSON
schemas, but the control loop is exactly the same. Guardrails implemented here:
max_steps, unknown-tool handling, tool exceptions surfaced as observations, and an
allow-list of tools.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Callable

from .llm import LLM


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


@dataclass
class AgentStep:
    thought: str
    action: str | None
    action_input: str | None
    observation: str | None


@dataclass
class AgentResult:
    answer: str | None
    steps: list[AgentStep] = field(default_factory=list)
    stopped_reason: str = "final_answer"


REACT_PROMPT = """Answer the question. You can use these tools:
{tools}

Use EXACTLY this format:
Thought: <reasoning>
Action: <tool name>[<input>]
(you will then receive) Observation: <result>
... (repeat Thought/Action/Observation as needed)
Thought: I now know the answer
Final Answer: <answer>

Question: {question}
{scratchpad}"""

_ACTION = re.compile(r"Action:\s*([\w\-]+)\[(.*?)\]\s*$", re.S | re.M)
_FINAL = re.compile(r"Final Answer:\s*(.*)", re.S)
_THOUGHT = re.compile(r"Thought:\s*(.*?)(?:\n|$)")


class ReActAgent:
    def __init__(self, llm: LLM, tools: list[Tool], max_steps: int = 6):
        self.llm, self.max_steps = llm, max_steps
        self.tools = {t.name: t for t in tools}

    def _prompt(self, question: str, scratchpad: str) -> str:
        tools = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        return REACT_PROMPT.format(tools=tools, question=question, scratchpad=scratchpad)

    def run(self, question: str) -> AgentResult:
        scratchpad, steps = "", []
        for _ in range(self.max_steps):
            out = self.llm.complete(self._prompt(question, scratchpad))
            thought = (_THOUGHT.search(out).group(1).strip() if _THOUGHT.search(out) else "")
            final = _FINAL.search(out)
            if final:
                steps.append(AgentStep(thought, None, None, None))
                return AgentResult(final.group(1).strip(), steps)
            act = _ACTION.search(out)
            if not act:
                obs = "Error: could not parse an Action or Final Answer. Follow the format."
                steps.append(AgentStep(thought, None, None, obs))
            else:
                name, arg = act.group(1), act.group(2)
                tool = self.tools.get(name)
                if tool is None:
                    obs = f"Error: unknown tool '{name}'. Available: {', '.join(self.tools)}"
                else:
                    try:
                        obs = str(tool.func(arg))
                    except Exception as e:  # tool errors become observations, not crashes
                        obs = f"Error: {type(e).__name__}: {e}"
                steps.append(AgentStep(thought, name, arg, obs))
            scratchpad += f"{out.strip()}\nObservation: {steps[-1].observation}\n"
        return AgentResult(None, steps, stopped_reason="max_steps")


# ---------------- example tools ----------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}


def safe_calculator(expr: str) -> str:
    """Evaluate arithmetic WITHOUT eval() — never give an LLM an unrestricted eval/exec tool."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    result = ev(ast.parse(expr.strip(), mode="eval").body)
    return str(int(result) if float(result).is_integer() else round(result, 6))


def make_lookup_tool(table: dict[str, str], name: str = "lookup", description: str = "Look up a key") -> Tool:
    return Tool(name, description, lambda k: table.get(k.strip().lower(), f"No entry for '{k}'"))


def make_sql_tool(conn, read_only: bool = True) -> Tool:
    """Text-to-SQL tool over a sqlite3 connection, with a read-only guardrail."""
    def run(sql: str) -> str:
        if read_only and not re.match(r"^\s*(select|with)\b", sql, re.I):
            raise PermissionError("only SELECT queries are allowed")
        if read_only and re.search(r";\s*\S", sql):
            raise PermissionError("multiple statements are not allowed")
        rows = conn.execute(sql).fetchall()
        return str(rows[:50])
    return Tool("sql", "Run a read-only SQL SELECT against the warehouse", run)
