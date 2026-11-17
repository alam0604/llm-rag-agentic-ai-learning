"""Vector stores: store embeddings + metadata, and find the nearest neighbours fast.

FlatIndex  — exact search, compares the query with every vector: O(N·d).
IVFIndex   — approximate search (Inverted File): k-means clusters the vectors,
             and at query time only the `n_probe` closest clusters are scanned.
             This is the idea behind FAISS IVF, pgvector ivfflat, Milvus IVF_*.

Production vector DBs (Pinecone, Weaviate, Qdrant, Milvus, Chroma, pgvector,
Azure AI Search, Databricks Vector Search) add HNSW graphs, quantization,
persistence, replication, metadata filtering and hybrid search on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .embeddings import l2_normalize


@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


MetadataFilter = Callable[[dict[str, Any]], bool]


class FlatIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim))
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.metas: list[dict] = []

    def __len__(self):
        return len(self.ids)

    def add(self, ids, vectors, texts=None, metadatas=None):
        vectors = l2_normalize(np.atleast_2d(np.asarray(vectors, dtype=float)))
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors length mismatch")
        dup = set(ids) & set(self.ids)
        if dup:
            raise ValueError(f"duplicate ids: {sorted(dup)}")
        self.vectors = np.vstack([self.vectors, vectors])
        self.ids += list(ids)
        self.texts += list(texts or [""] * len(ids))
        self.metas += list(metadatas or [{} for _ in ids])

    def delete(self, id_: str):
        i = self.ids.index(id_)
        self.vectors = np.delete(self.vectors, i, axis=0)
        for lst in (self.ids, self.texts, self.metas):
            lst.pop(i)

    def _candidates(self, query: np.ndarray) -> np.ndarray:
        return np.arange(len(self.ids))

    def search(self, query, k: int = 3, where: MetadataFilter | None = None) -> list[SearchResult]:
        if not self.ids:
            return []
        q = l2_normalize(np.asarray(query, dtype=float).reshape(1, -1))[0]
        cand = self._candidates(q)
        if where is not None:  # pre-filtering on metadata
            cand = np.array([i for i in cand if where(self.metas[i])], dtype=int)
        if len(cand) == 0:
            return []
        scores = self.vectors[cand] @ q  # cosine (vectors are normalized)
        top = np.argsort(-scores)[:k]
        return [SearchResult(self.ids[cand[i]], float(scores[i]), self.texts[cand[i]], self.metas[cand[i]])
                for i in top]


def kmeans(x: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = min(k, len(x))
    xn = l2_normalize(x)
    # k-means++ style init: spread initial centroids out (picks far-apart points)
    idx = [int(rng.integers(len(x)))]
    for _ in range(1, k):
        sim = np.max(xn @ xn[idx].T, axis=1)
        idx.append(int(np.argmin(sim)))
    centroids = x[idx].copy()
    for _ in range(iters):
        assign = np.argmax(l2_normalize(x) @ l2_normalize(centroids).T, axis=1)
        for c in range(k):
            members = x[assign == c]
            if len(members):
                centroids[c] = members.mean(0)
    return l2_normalize(centroids), assign


class IVFIndex(FlatIndex):
    """Approximate nearest neighbour search with an inverted file of clusters."""

    def __init__(self, dim: int, n_lists: int = 8, n_probe: int = 2):
        super().__init__(dim)
        self.n_lists, self.n_probe = n_lists, n_probe
        self.centroids: np.ndarray | None = None
        self.assign: np.ndarray | None = None

    def train(self):
        self.centroids, self.assign = kmeans(self.vectors, self.n_lists)

    def add(self, *a, **kw):
        super().add(*a, **kw)
        self.train()  # (re)train for simplicity; real systems train once on a sample

    def _candidates(self, q):
        if self.centroids is None:
            return super()._candidates(q)
        nearest = np.argsort(-(self.centroids @ q))[: self.n_probe]
        return np.where(np.isin(self.assign, nearest))[0]
