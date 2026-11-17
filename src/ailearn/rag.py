"""An end-to-end Retrieval-Augmented Generation pipeline.

    INDEXING (offline)                 QUERYING (online)
    docs -> chunk -> embed -> store    question -> embed -> retrieve top-k
                                                -> (optional) hybrid + rerank
                                                -> build prompt with context
                                                -> LLM -> answer + citations
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .bm25 import BM25
from .chunking import recursive_chunks
from .embeddings import Embedder, HashingEmbedder
from .hybrid import reciprocal_rank_fusion
from .llm import LLM, MockLLM
from .vectorstore import FlatIndex, SearchResult

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided context."
Cite the sources you used with their [number].

<context>
{context}
</context>

<question>{question}</question>
"""


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    prompt: str = ""


def build_prompt(question: str, results: list[SearchResult], template: str = PROMPT_TEMPLATE) -> str:
    context = "\n".join(f"[{i}] {r.text}" for i, r in enumerate(results, start=1))
    return template.format(context=context, question=question)


class RAGPipeline:
    def __init__(self, embedder: Embedder | None = None, llm: LLM | None = None,
                 chunk_size: int = 400, top_k: int = 3, hybrid: bool = False, min_score: float = 0.0):
        self.embedder = embedder or HashingEmbedder()
        self.llm = llm or MockLLM()
        self.chunk_size, self.top_k, self.hybrid, self.min_score = chunk_size, top_k, hybrid, min_score
        self.index: FlatIndex | None = None
        self._bm25: BM25 | None = None

    # ---------- indexing ----------
    def add_documents(self, docs: dict[str, str], metadata: dict[str, dict] | None = None) -> int:
        ids, texts, metas = [], [], []
        for doc_id, text in docs.items():
            for j, chunk in enumerate(recursive_chunks(text, self.chunk_size)):
                ids.append(f"{doc_id}#{j}")
                texts.append(chunk)
                metas.append({"doc_id": doc_id, "chunk": j, **(metadata or {}).get(doc_id, {})})
        vecs = self.embedder.embed(texts)
        if self.index is None:
            self.index = FlatIndex(vecs.shape[1])
        self.index.add(ids, vecs, texts, metas)
        self._bm25 = BM25(self.index.texts)
        return len(ids)

    # ---------- retrieval ----------
    def retrieve(self, question: str, k: int | None = None, where=None) -> list[SearchResult]:
        k = k or self.top_k
        if self.index is None:
            return []
        q = self.embedder.embed([question])[0]
        dense = [r for r in self.index.search(q, k=k * 3, where=where) if r.score > self.min_score]
        if not self.hybrid:
            return dense[:k]
        sparse_ids = [self.index.ids[i] for i, _ in self._bm25.search(question, k=k * 3)]
        if where is not None:
            sparse_ids = [i for i in sparse_ids if where(self.index.metas[self.index.ids.index(i)])]
        fused = reciprocal_rank_fusion([[r.id for r in dense], sparse_ids], top_n=k)
        out = []
        for doc_id, score in fused:
            i = self.index.ids.index(doc_id)
            out.append(SearchResult(doc_id, score, self.index.texts[i], self.index.metas[i]))
        return out

    # ---------- generation ----------
    def ask(self, question: str, k: int | None = None, where=None) -> RAGAnswer:
        results = self.retrieve(question, k, where)
        if not results:
            return RAGAnswer(question, "I don't know based on the provided context.", [], "")
        prompt = build_prompt(question, results)
        return RAGAnswer(question, self.llm.complete(prompt), results, prompt)
