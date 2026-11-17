"""Evaluating retrieval and generation.

"You can't improve what you don't measure." Build a small golden dataset
(question, relevant chunk ids, reference answer) and track these metrics every
time you change chunking, embeddings, k, prompts, or models.

Retrieval : precision@k, recall@k, hit_rate@k, MRR, nDCG@k
Generation: groundedness (is the answer supported by context?),
            answer_correctness (token F1 vs reference), and LLM-as-judge (see docs).
"""
from __future__ import annotations

import math
from collections import Counter

from .embeddings import tokenize


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return sum(d in relevant for d in top) / k if k else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevance: dict[str, float], k: int) -> float:
    dcg = sum(relevance.get(d, 0) / math.log2(i + 1) for i, d in enumerate(retrieved[:k], start=1))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def token_f1(prediction: str, reference: str) -> float:
    p, r = tokenize(prediction), tokenize(reference)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def groundedness(answer: str, contexts: list[str], threshold: float = 0.6) -> float:
    """Fraction of answer sentences whose content words mostly appear in the context.

    A cheap lexical proxy for 'faithfulness'. Production systems use NLI models or
    an LLM judge (RAGAS, TruLens, DeepEval) — but the idea is the same.
    """
    import re
    ctx = set(tokenize(" ".join(contexts)))
    sents = [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\[\d+\]", "", answer)) if tokenize(s)]
    if not sents:
        return 0.0
    supported = sum(len(set(tokenize(s)) & ctx) / len(set(tokenize(s))) >= threshold for s in sents)
    return supported / len(sents)


def evaluate_retriever(retrieve_fn, dataset: list[dict], k: int = 3) -> dict[str, float]:
    """dataset items: {"question": str, "relevant": set[str]} ; retrieve_fn(q, k) -> list[id]."""
    agg = Counter()
    for item in dataset:
        got = retrieve_fn(item["question"], k)
        rel = set(item["relevant"])
        agg["precision@k"] += precision_at_k(got, rel, k)
        agg["recall@k"] += recall_at_k(got, rel, k)
        agg["hit_rate@k"] += hit_rate_at_k(got, rel, k)
        agg["mrr"] += mrr(got, rel)
    n = len(dataset) or 1
    return {m: round(v / n, 4) for m, v in agg.items()}
