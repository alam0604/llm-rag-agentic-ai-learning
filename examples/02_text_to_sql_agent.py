"""A data-engineering agent: answers business questions by writing SQL against a warehouse.

Offline by default (ScriptedLLM shows what a real model would emit).
Set USE_REAL_LLM=1 and OPENAI_API_KEY (or point base_url at Ollama) to use a real model.

    python examples/02_text_to_sql_agent.py
"""
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ailearn.agent import ReActAgent, Tool, make_sql_tool, safe_calculator  # noqa: E402
from ailearn.llm import ScriptedLLM  # noqa: E402

conn = sqlite3.connect(":memory:")
conn.executescript("""
create table gold_daily_sales(sale_date text, region text, revenue real, orders int);
insert into gold_daily_sales values
 ('2026-11-01','north',12000,120),('2026-11-01','south',8000,95),
 ('2026-11-02','north',13500,130),('2026-11-02','south',9100,101);
""")

schema = Tool("schema", "Show table names and columns",
              lambda _: str(conn.execute("select sql from sqlite_master").fetchall()))
tools = [schema, make_sql_tool(conn), Tool("calculator", "Arithmetic", safe_calculator)]

if os.getenv("USE_REAL_LLM"):
    from ailearn.llm import OpenAICompatibleLLM
    llm = OpenAICompatibleLLM()
else:
    llm = ScriptedLLM([
        "Thought: I should inspect the schema first.\nAction: schema[]",
        "Thought: Sum revenue by region.\n"
        "Action: sql[select region, sum(revenue) from gold_daily_sales group by region order by 2 desc]",
        "Thought: north=25500, south=17100. Compute the growth gap percentage.\nAction: calculator[(25500-17100)/17100*100]",
        "Thought: I now know the answer\nFinal Answer: North leads with 25,500 revenue vs South 17,100 (about 49% higher).",
    ])

result = ReActAgent(llm, tools).run("Which region has the highest revenue and by how much?")
for i, s in enumerate(result.steps, 1):
    print(f"Step {i}: {s.thought}\n   Action: {s.action}[{s.action_input}]\n   Observation: {s.observation}")
print("\nFINAL:", result.answer)
