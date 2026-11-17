# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Build a Vector Search index from a Delta table (Mosaic AI Vector Search)
# MAGIC Run on a Databricks workspace with Unity Catalog + Vector Search enabled.
# MAGIC Pipeline: Volume files → silver text → gold chunks (Delta, CDF on) → Delta Sync index.
# MAGIC > Databricks APIs evolve — check the docs linked in `docs/10-hands-on-langgraph-databricks.md` if a call changes.

# COMMAND ----------
# MAGIC %pip install -q databricks-vectorsearch pypdf
# MAGIC %restart_python

# COMMAND ----------
CATALOG, SCHEMA = "main", "rag_demo"
SOURCE_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw_docs"         # upload PDFs/MD here
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.gold_doc_chunks"
VS_ENDPOINT = "rag_demo_endpoint"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.doc_chunks_index"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"                  # a Foundation Model API embedding endpoint

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw_docs")

# COMMAND ----------
# Silver: parse files to text
import io
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

@F.pandas_udf(StringType())
def parse_doc(content: pd.Series, path: pd.Series) -> pd.Series:
    from pypdf import PdfReader
    def to_text(b, p):
        if p.lower().endswith(".pdf"):
            return "\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(b)).pages)
        return b.decode("utf-8", errors="ignore")
    return pd.Series([to_text(b, p) for b, p in zip(content, path)])

silver = (spark.read.format("binaryFile").load(SOURCE_VOLUME)
          .withColumn("text", parse_doc("content", "path"))
          .select("path", "modificationTime", "text"))

# COMMAND ----------
# Gold: chunks with deterministic ids (idempotent upserts)
@F.pandas_udf(ArrayType(StringType()))
def chunk(text: pd.Series) -> pd.Series:
    def split(t, size=1500, overlap=200):
        t = t or ""
        return [t[i:i + size] for i in range(0, max(len(t) - overlap, 1), size - overlap) if t[i:i + size].strip()]
    return text.apply(split)

gold = (silver.withColumn("chunk", F.explode(chunk("text")))
        .withColumn("chunk_id", F.sha2(F.concat_ws("|", "path", "chunk"), 256))
        .select("chunk_id", "path", "modificationTime", "chunk"))

(gold.write.mode("overwrite").option("overwriteSchema", "true")
     .option("delta.enableChangeDataFeed", "true").saveAsTable(CHUNKS_TABLE))
display(spark.table(CHUNKS_TABLE).limit(5))

# COMMAND ----------
# Vector Search endpoint + Delta Sync index (Databricks computes the embeddings)
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

if VS_ENDPOINT not in [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")

vsc.create_delta_sync_index(
    endpoint_name=VS_ENDPOINT,
    index_name=INDEX_NAME,
    source_table_name=CHUNKS_TABLE,
    pipeline_type="TRIGGERED",                 # re-sync after each ingestion run (or CONTINUOUS)
    primary_key="chunk_id",
    embedding_source_column="chunk",
    embedding_model_endpoint_name=EMBEDDING_ENDPOINT,
)

# COMMAND ----------
# Query it (wait until the index is ONLINE)
idx = vsc.get_index(VS_ENDPOINT, INDEX_NAME)
idx.similarity_search(query_text="How do I fix copy activity error 2200?",
                      columns=["chunk_id", "path", "chunk"], num_results=3, query_type="HYBRID")

# COMMAND ----------
# After the next ingestion run: idx.sync()  → only changed rows (CDF) are re-embedded
