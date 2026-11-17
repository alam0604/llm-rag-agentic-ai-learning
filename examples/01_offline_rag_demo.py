"""Offline RAG demo — no API key needed.

    python examples/01_offline_rag_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ailearn.rag import RAGPipeline  # noqa: E402

DOCS = {
    "runbook_adf": "If an ADF copy activity fails with error 2200, check the linked service credentials. "
                   "Error 2200 is a user configuration error on the source or sink. "
                   "Retry the pipeline only after fixing the connection string.",
    "delta_guide": "Run OPTIMIZE on Delta tables weekly to compact small files. "
                   "Use VACUUM to remove files older than the retention period, default 7 days. "
                   "Time travel lets you query a previous version with VERSION AS OF.",
    "oncall": "The data platform on-call rotation changes every Monday. "
              "Escalate P1 incidents to the platform lead within 15 minutes.",
}

rag = RAGPipeline(top_k=2, hybrid=True)
print("indexed chunks:", rag.add_documents(DOCS))

for q in ["What does error 2200 mean?", "How long is the default VACUUM retention?",
          "When does the on-call rotation change?", "What is the capital of France?"]:
    ans = rag.ask(q)
    print(f"\nQ: {q}\nA: {ans.answer}")
    for i, s in enumerate(ans.sources, 1):
        print(f"   [{i}] {s.id}  score={s.score:.3f}")
