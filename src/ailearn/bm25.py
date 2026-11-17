"""BM25 — the keyword (sparse) retrieval baseline used by Elasticsearch/OpenSearch/Lucene.

score(D, Q) = Σ_{q in Q} IDF(q) · f(q,D)·(k1+1) / (f(q,D) + k1·(1 - b + b·|D|/avgdl))

Dense embeddings capture *meaning*; BM25 captures *exact terms* (error codes, IDs,
product names, table names like `fact_sales_daily`). Hybrid search combines both.
"""
from __future__ import annotations

import math
from collections import Counter

from .embeddings import tokenize


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in corpus]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str, i: int) -> float:
        tf, dl, s = self.tf[i], len(self.docs[i]), 0.0
        for q in tokenize(query):
            if q not in tf:
                continue
            f = tf[q]
            s += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scores = [x for x in scores if x[1] > 0]
        return sorted(scores, key=lambda x: -x[1])[:k]
