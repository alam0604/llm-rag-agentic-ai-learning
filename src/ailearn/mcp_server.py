"""An MCP server that exposes a SQL warehouse + docs search to any MCP client
(Claude apps, IDE agents, LangGraph via MCP adapters, your own agent).

    pip install "mcp[cli]"          # this file targets the MCP Python SDK 2.x (MCPServer)
    python -m ailearn.mcp_server    # stdio transport (what desktop/IDE clients launch)
    WAREHOUSE_DB=warehouse.db python -m ailearn.mcp_server --http   # streamable HTTP on :8000

SDK 1.x used `from mcp.server.fastmcp import FastMCP`; 2.x renamed it to `MCPServer`.

Exposes
  tools     : list_tables, describe_table, run_readonly_query, search_docs
  resources : warehouse://schema
  prompts   : analyze_table
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys

from mcp.server.mcpserver import MCPServer

from .graph_agent import validate_readonly_sql
from .rag import RAGPipeline

MAX_ROWS = 100
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def build_server(conn: sqlite3.Connection, rag: RAGPipeline | None = None, name: str = "data-platform") -> MCPServer:
    server = MCPServer(name, instructions=(
        "Tools for a read-only analytics warehouse and data-platform docs. "
        "Call list_tables/describe_table before writing SQL. Queries are SELECT-only and capped at 100 rows."))

    def _tables() -> list[str]:
        return [r[0] for r in conn.execute("select name from sqlite_master where type='table' order by name")]

    @server.tool(description="List all tables in the warehouse.")
    def list_tables() -> list[str]:
        return _tables()

    @server.tool(description="Show the columns (name, type) of one table.")
    def describe_table(table: str) -> list[dict]:
        if not _IDENT.match(table) or table not in _tables():
            raise ValueError(f"Unknown table '{table}'. Call list_tables first.")
        return [{"column": r[1], "type": r[2]} for r in conn.execute(f"pragma table_info({table})")]

    @server.tool(description="Run ONE read-only SQL SELECT. Returns at most 100 rows as a list of objects.")
    def run_readonly_query(sql: str) -> list[dict]:
        err = validate_readonly_sql(sql)
        if err:
            raise ValueError(err)
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchmany(MAX_ROWS)]

    @server.tool(description="Semantic + keyword search over data-platform runbooks and docs.")
    def search_docs(query: str, k: int = 3) -> list[dict]:
        if rag is None:
            return []
        return [{"id": r.id, "score": round(r.score, 4), "text": r.text} for r in rag.retrieve(query, k=k)]

    @server.resource("warehouse://schema", description="CREATE statements for every table", mime_type="text/plain")
    def schema() -> str:
        return "\n".join(r[0] for r in conn.execute("select sql from sqlite_master where type='table'"))

    @server.prompt(description="A prompt template that asks the model to profile a table.")
    def analyze_table(table: str) -> str:
        return (f"Profile the table `{table}`: call describe_table, then run_readonly_query to count rows, "
                f"nulls per column and min/max of numeric columns. Summarise data-quality issues.")

    return server


def _demo_conn() -> sqlite3.Connection:
    path = os.getenv("WAREHOUSE_DB")
    conn = sqlite3.connect(path or ":memory:", check_same_thread=False)
    if not path:
        conn.executescript("""
            create table gold_daily_sales(sale_date text, region text, revenue real);
            insert into gold_daily_sales values ('2026-11-01','north',12000),('2026-11-01','south',8000);
        """)
    return conn


if __name__ == "__main__":  # pragma: no cover
    srv = build_server(_demo_conn())
    srv.run("streamable-http" if "--http" in sys.argv else "stdio")
