"""A minimal Byte-Pair Encoding (BPE) tokenizer.

GPT-style models do not see words or characters — they see *tokens* produced by
an algorithm like this. BPE starts from bytes and repeatedly merges the most
frequent adjacent pair into a new token.

Why it matters to you as an engineer:
  * API pricing and context limits are measured in tokens, not words.
  * ~1 token ≈ 4 English characters ≈ 0.75 words (rule of thumb).
  * Rare words / code / non-English text use more tokens.
"""
from __future__ import annotations

from collections import Counter


def _pair_counts(ids: list[int]) -> Counter:
    return Counter(zip(ids, ids[1:]))


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (the byte alphabet)")
        ids = list(text.encode("utf-8"))
        for new_id in range(256, vocab_size):
            counts = _pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            if counts[pair] < 2:  # nothing worth merging
                break
            ids = _merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return self

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = _pair_counts(ids)
            # apply the earliest-learned merge first (lowest new id)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def tokens(self, text: str) -> list[str]:
        """Human-readable view of how text is split."""
        return [self.vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)]


def estimate_tokens(text: str) -> int:
    """Quick rule-of-thumb estimate (~4 chars/token for English)."""
    return max(1, round(len(text) / 4))
