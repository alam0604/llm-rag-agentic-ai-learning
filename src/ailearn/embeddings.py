"""Embeddings: turning text into vectors so that *similar meaning ≈ nearby vectors*.

Real systems use neural embedding models (e.g. sentence-transformers,
OpenAI text-embedding-3, Cohere embed, BGE, E5). Here we implement two classic,
offline embedders so tests are deterministic and you can see the mechanics:

* TfidfEmbedder    — fits a vocabulary; good to understand weighting.
* HashingEmbedder  — no fitting needed; fixed dimension via the hashing trick.

Swap in a neural model with the same `.embed(list[str]) -> np.ndarray` interface
(see `SentenceTransformerEmbedder`).
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "and", "or",
    "for", "with", "as", "by", "it", "this", "that", "be", "at", "from", "what", "how",
    "do", "does", "i", "you", "can", "which", "who",
}


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    return [t for t in toks if not (remove_stopwords and t in STOPWORDS)]


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of a (n,d) and rows of b (m,d) -> (n,m)."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    return l2_normalize(a) @ l2_normalize(b).T


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


def dot_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.atleast_2d(a) @ np.atleast_2d(b).T


class TfidfEmbedder:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, corpus: list[str]) -> "TfidfEmbedder":
        df = Counter()
        for doc in corpus:
            df.update(set(tokenize(doc)))
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        n = len(corpus)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)])
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("call fit() first")
        m = np.zeros((len(texts), len(self.vocab)))
        for r, text in enumerate(texts):
            for tok, c in Counter(tokenize(text)).items():
                if tok in self.vocab:
                    m[r, self.vocab[tok]] = c
        return l2_normalize(m * self.idf)


class HashingEmbedder:
    """Stateless embedder: hash each token (and bigram) into one of `dim` buckets."""

    def __init__(self, dim: int = 512, use_bigrams: bool = True):
        self.dim, self.use_bigrams = dim, use_bigrams

    def _bucket(self, tok: str) -> tuple[int, float]:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        return h % self.dim, (1.0 if (h >> 64) & 1 else -1.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim))
        for r, text in enumerate(texts):
            toks = tokenize(text)
            feats = toks + ([f"{a}_{b}" for a, b in zip(toks, toks[1:])] if self.use_bigrams else [])
            for f in feats:
                idx, sign = self._bucket(f)
                m[r, idx] += sign
        return l2_normalize(m)


class SentenceTransformerEmbedder:  # pragma: no cover - optional heavy dependency
    """Real neural embeddings. `pip install sentence-transformers`."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))
