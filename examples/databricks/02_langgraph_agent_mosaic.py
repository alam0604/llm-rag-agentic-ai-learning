# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · LangGraph agent on Databricks: tools, MLflow tracing, evaluation, deployment
# MAGIC Tools: Vector Search retriever (docs) + a Unity Catalog SQL function (numbers).
# MAGIC LLM: a Foundation Model API endpoint. Tracing/eval: MLflow 3. Deploy: Agent Framework.
# MAGIC > Package/class names in this area change often — confirm against the docs linked in lesson 10.

# COMMAND ----------
# MAGIC %pip install -q -U langgraph databricks-langchain databricks-agents "mlflow>=3" unitycatalog-ai[databricks]
# MAGIC %restart_python

# COMMAND ----------
CATALOG, SCHEMA = "main", "rag_demo"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"          # any chat endpoint you have
INDEX_NAME = f"{CATALOG}.{SCHEMA}.doc_chunks_index"
UC_MODEL = f"{CATALOG}.{SCHEMA}.data_platform_agent"

# COMMAND ----------
# A governed SQL tool as a Unity Catalog function (the agent can only do what this function does)
spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.revenue_by_region(days INT COMMENT 'look-back window in days')
RETURNS TABLE (region STRING, revenue DOUBLE)
COMMENT 'Total revenue per region over the last N days from gold_daily_sales'
RETURN SELECT region, SUM(revenue) FROM {CATALOG}.{SCHEMA}.gold_daily_sales
       WHERE sale_date >= date_sub(current_date(), days) GROUP BY region
""")

# COMMAND ----------
import mlflow
from databricks_langchain import ChatDatabricks, UCFunctionToolkit, VectorSearchRetrieverTool
from langgraph.prebuilt import create_react_agent

mlflow.langchain.autolog()                      # every LLM + tool call becomes an MLflow trace

llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0)
docs_tool = VectorSearchRetrieverTool(
    index_name=INDEX_NAME, num_results=4,
    tool_name="search_runbooks",
    tool_description="Search data-platform runbooks and docs. Use for how-to, errors and concepts.")
sql_tools = UCFunctionToolkit(function_names=[f"{CATALOG}.{SCHEMA}.revenue_by_region"]).tools

SYSTEM = ("You are a data-platform assistant. Use search_runbooks for how-to/error questions and the "
          "revenue tool for numbers. Cite runbook paths. If tools don't have the answer, say so.")
agent = create_react_agent(llm, tools=[docs_tool, *sql_tools], prompt=SYSTEM)

# COMMAND ----------
agent.invoke({"messages": [{"role": "user", "content": "Which region made the most revenue in the last 7 days?"}]})

# COMMAND ----------
# MAGIC %md ## Evaluate with MLflow (LLM judges + your golden set)

# COMMAND ----------
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety, RetrievalGroundedness

eval_data = [
    {"inputs": {"question": "What does ADF error 2200 mean?"},
     "expectations": {"expected_facts": ["user configuration problem", "source or sink"]}},
    {"inputs": {"question": "How often should OPTIMIZE run on Delta tables?"},
     "expectations": {"expected_facts": ["weekly"]}},
]

def predict_fn(question: str) -> str:
    out = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return out["messages"][-1].content

results = mlflow.genai.evaluate(data=eval_data, predict_fn=predict_fn,
                                scorers=[Correctness(), RelevanceToQuery(), Safety(), RetrievalGroundedness()])
results.metrics

# COMMAND ----------
# MAGIC %md
# MAGIC ## Log, register and deploy
# MAGIC Put the agent in `agent.py` (models-from-code), wrap it in `mlflow.pyfunc.ResponsesAgent`
# MAGIC (see the Agent Framework "author an agent" docs), then:
# MAGIC ```python
# MAGIC from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex, DatabricksFunction
# MAGIC with mlflow.start_run():
# MAGIC     info = mlflow.pyfunc.log_model(
# MAGIC         name="agent", python_model="agent.py",
# MAGIC         resources=[DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
# MAGIC                    DatabricksVectorSearchIndex(index_name=INDEX_NAME),
# MAGIC                    DatabricksFunction(function_name=f"{CATALOG}.{SCHEMA}.revenue_by_region")])
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC mv = mlflow.register_model(info.model_uri, UC_MODEL)
# MAGIC from databricks import agents
# MAGIC agents.deploy(UC_MODEL, mv.version)      # serving endpoint + Review App for human feedback
# MAGIC ```
