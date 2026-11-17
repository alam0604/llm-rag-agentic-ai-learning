"""Chunking: splitting documents into retrievable pieces.

Chunk size is one of the highest-leverage knobs in RAG:
  * too small  -> chunks lose context ("it" no longer refers to anything)
  * too large  -> embeddings get blurry, and you waste context window / money
Typical starting point: 300-800 tokens with 10-20% overlap, then tune with evals.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


def fixed_size_chunks(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Character windows with overlap. Simple and predictable, but cuts mid-sentence."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step) if text[i:i + size].strip()]


_SENT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def sentence_chunks(text: str, max_chars: int = 300, overlap_sentences: int = 1) -> list[str]:
    """Pack whole sentences into chunks up to max_chars, repeating the last N sentences."""
    sents = split_sentences(text)
    chunks, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s])) > max_chars:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def recursive_chunks(text: str, max_chars: int = 300,
                     separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ")) -> list[str]:
    """LangChain-style recursive splitter: try paragraph, then line, sentence, word."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    for sep in separators:
        if sep in text:
            parts, out, buf = text.split(sep), [], ""
            for p in parts:
                cand = f"{buf}{sep}{p}" if buf else p
                if len(cand) <= max_chars:
                    buf = cand
                else:
                    if buf:
                        out.append(buf)
                    if len(p) > max_chars:
                        out.extend(recursive_chunks(p, max_chars, separators))
                        buf = ""
                    else:
                        buf = p
            if buf:
                out.append(buf)
            return [c.strip() for c in out if c.strip()]
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def markdown_section_chunks(md: str) -> list[Chunk]:
    """Structure-aware: one chunk per markdown heading, keeping the heading path as metadata."""
    chunks, path, buf = [], [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            chunks.append(Chunk(body, len(chunks), {"section": " > ".join(path)}))

    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return chunks
