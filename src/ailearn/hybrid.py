"""Hybrid retrieval with Reciprocal Rank Fusion (RRF).

RRF(d) = Σ_r 1 / (k + rank_r(d))   (k≈60)

RRF only uses *ranks*, so you can fuse lists whose scores are on different
scales (cosine similarity vs BM25) without any normalisation.
"""
from __future__ import annotations


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60, top_n: int | None = None) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return fused[:top_n] if top_n else fused


def weighted_score_fusion(dense: dict[str, float], sparse: dict[str, float], alpha: float = 0.5) -> list[tuple[str, float]]:
    """Alternative: min-max normalise both score sets, then alpha*dense + (1-alpha)*sparse."""
    def norm(d):
        if not d:
            return {}
        lo, hi = min(d.values()), max(d.values())
        return {k: (v - lo) / (hi - lo) if hi > lo else 1.0 for k, v in d.items()}
    nd, ns = norm(dense), norm(sparse)
    keys = set(nd) | set(ns)
    out = {key: alpha * nd.get(key, 0) + (1 - alpha) * ns.get(key, 0) for key in keys}
    return sorted(out.items(), key=lambda x: (-x[1], x[0]))
