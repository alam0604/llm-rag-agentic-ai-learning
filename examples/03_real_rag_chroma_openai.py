"""A *real* RAG app: sentence-transformers embeddings + ChromaDB + any OpenAI-compatible LLM.

pip install chromadb sentence-transformers openai
export OPENAI_API_KEY=...            # or run Ollama locally (free):
# ollama pull llama3.1 && export OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama MODEL=llama3.1

    python examples/03_real_rag_chroma_openai.py ./my_docs_folder "your question"
"""
import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ailearn.chunking import recursive_chunks  # noqa: E402

folder, question = sys.argv[1], sys.argv[2]

client = chromadb.PersistentClient(path=".chroma")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
col = client.get_or_create_collection("docs", embedding_function=ef, metadata={"hnsw:space": "cosine"})

# ---- index (idempotent via upsert) ----
for path in Path(folder).rglob("*"):
    if path.suffix.lower() in {".md", ".txt", ".sql", ".py"}:
        chunks = recursive_chunks(path.read_text(errors="ignore"), max_chars=1200)
        col.upsert(ids=[f"{path}#{i}" for i in range(len(chunks))], documents=chunks,
                   metadatas=[{"source": str(path), "chunk": i} for i in range(len(chunks))])

# ---- retrieve ----
res = col.query(query_texts=[question], n_results=4)
context = "\n\n".join(f"[{i+1}] (source: {m['source']})\n{d}"
                      for i, (d, m) in enumerate(zip(res["documents"][0], res["metadatas"][0])))

# ---- generate ----
llm = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))
msg = llm.chat.completions.create(
    model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0,
    messages=[
        {"role": "system", "content": "Answer ONLY from the context. Cite sources like [1]. "
                                      "If the answer is not in the context, say you don't know."},
        {"role": "user", "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}"},
    ])
print(msg.choices[0].message.content)
