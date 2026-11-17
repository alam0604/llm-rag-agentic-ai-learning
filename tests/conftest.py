import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# A tiny knowledge base used across tests (data-engineering flavoured).
DOCS = {
    "delta": (
        "Delta Lake is an open-source storage layer that brings ACID transactions to data lakes. "
        "Delta tables support time travel, so you can query older versions of data with VERSION AS OF. "
        "The OPTIMIZE command compacts small files and ZORDER BY co-locates related data."
    ),
    "medallion": (
        "The medallion architecture organises a lakehouse into bronze, silver and gold layers. "
        "Bronze holds raw ingested data. Silver holds cleaned and conformed data. "
        "Gold holds business-level aggregates ready for reporting."
    ),
    "adf": (
        "Azure Data Factory is a cloud ETL and orchestration service. "
        "Pipelines in ADF contain activities such as Copy Data and Databricks Notebook. "
        "Triggers can schedule pipelines or start them when a blob arrives in storage."
    ),
    "rag": (
        "Retrieval-Augmented Generation retrieves relevant documents and adds them to the prompt. "
        "RAG reduces hallucination because the model answers from supplied context. "
        "Chunking, embeddings and a vector database are the core RAG components."
    ),
    "errors": (
        "Error code DF-SYS-01 in a mapping data flow usually means a cluster or spark failure. "
        "Error code 2200 in a copy activity means a user configuration problem with the source or sink."
    ),
}


@pytest.fixture
def docs():
    return dict(DOCS)
