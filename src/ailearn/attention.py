"""Self-attention from scratch (NumPy).

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V

This is the single most important equation in modern LLMs. Read docs/01-foundations.md
alongside this file.
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (subtract the max before exp)."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_mask(seq_len: int) -> np.ndarray:
    """Upper-triangular mask so token i can only attend to tokens <= i.

    Returns a (seq_len, seq_len) matrix with 0 where allowed and -inf where blocked.
    This is what makes GPT-style models *autoregressive*.
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return np.where(mask == 1, -np.inf, 0.0)


def scaled_dot_product_attention(q, k, v, mask=None):
    """q, k: (..., seq, d_k); v: (..., seq, d_v). Returns (output, weights)."""
    d_k = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


class MultiHeadSelfAttention:
    """Multi-head self-attention with random (untrained) projection weights.

    Shapes: input x is (seq_len, d_model). d_model must be divisible by n_heads.
    """

    def __init__(self, d_model: int, n_heads: int, seed: int = 0):
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        rng = np.random.default_rng(seed)
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        scale = 1 / np.sqrt(d_model)
        self.w_q = rng.normal(0, scale, (d_model, d_model))
        self.w_k = rng.normal(0, scale, (d_model, d_model))
        self.w_v = rng.normal(0, scale, (d_model, d_model))
        self.w_o = rng.normal(0, scale, (d_model, d_model))

    def _split(self, x):  # (seq, d_model) -> (heads, seq, d_head)
        seq = x.shape[0]
        return x.reshape(seq, self.n_heads, self.d_head).transpose(1, 0, 2)

    def __call__(self, x: np.ndarray, causal: bool = True):
        q, k, v = (self._split(x @ w) for w in (self.w_q, self.w_k, self.w_v))
        mask = causal_mask(x.shape[0]) if causal else None
        out, weights = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 0, 2).reshape(x.shape[0], self.d_model)  # concat heads
        return out @ self.w_o, weights


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Positional encoding from 'Attention Is All You Need' (2017)."""
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


def sample_next_token(logits: np.ndarray, temperature: float = 1.0, top_k: int | None = None,
                      top_p: float | None = None, rng: np.random.Generator | None = None) -> int:
    """Decoding strategies in one function: greedy (temperature=0), temperature, top-k, top-p."""
    rng = rng or np.random.default_rng()
    logits = np.asarray(logits, dtype=float)
    if temperature == 0:
        return int(np.argmax(logits))
    probs = softmax(logits / temperature)
    if top_k is not None:
        cutoff = np.sort(probs)[-top_k]
        probs = np.where(probs >= cutoff, probs, 0)
    if top_p is not None:
        order = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[order])
        keep = order[: int(np.searchsorted(cum, top_p) + 1)]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        probs = filtered
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))
