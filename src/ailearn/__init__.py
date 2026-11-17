"""ailearn — tiny, dependency-light implementations of the core building blocks
behind LLMs, RAG and agents. Everything runs offline (no API keys) so you can
read the code, run the tests, and *see* how each piece works.

Modules
-------
attention   : scaled dot-product + multi-head self-attention in NumPy
tokenizer   : a minimal Byte-Pair-Encoding (BPE) tokenizer
embeddings  : TF-IDF / hashing embeddings + cosine similarity
vectorstore : brute-force (exact) and IVF (approximate) vector indexes
chunking    : fixed-size, sentence-aware, and recursive text splitters
bm25        : classic keyword ranking (sparse retrieval)
hybrid      : Reciprocal Rank Fusion (RRF) of dense + sparse results
rag         : an end-to-end RAG pipeline with a pluggable LLM
agent       : a ReAct-style tool-using agent with a scripted/mock LLM
evaluation  : retrieval metrics (recall@k, precision@k, MRR, nDCG) + groundedness
llm         : LLM interface, a deterministic MockLLM, and optional real-API adapters
"""

__version__ = "1.0.0"
