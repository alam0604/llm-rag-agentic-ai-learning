import numpy as np
import pytest

from ailearn.chunking import (fixed_size_chunks, markdown_section_chunks, recursive_chunks,
                              sentence_chunks, split_sentences)
from ailearn.embeddings import HashingEmbedder
from ailearn.vectorstore import FlatIndex, IVFIndex


@pytest.fixture
def index(docs):
    emb = HashingEmbedder(dim=256)
    idx = FlatIndex(256)
    idx.add(list(docs), emb.embed(list(docs.values())), list(docs.values()),
            [{"team": "platform" if k in {"delta", "adf"} else "ai"} for k in docs])
    return idx, emb


def test_flat_search_returns_most_similar_first(index):
    idx, emb = index
    res = idx.search(emb.embed(["bronze silver gold layers"])[0], k=2)
    assert res[0].id == "medallion"
    assert res[0].score >= res[1].score


def test_metadata_filter(index):
    idx, emb = index
    res = idx.search(emb.embed(["bronze silver gold layers"])[0], k=5, where=lambda m: m["team"] == "platform")
    assert {r.id for r in res} <= {"delta", "adf"}


def test_add_validation_and_delete(index):
    idx, emb = index
    with pytest.raises(ValueError):
        idx.add(["delta"], emb.embed(["dup"]))          # duplicate id
    with pytest.raises(ValueError):
        idx.add(["x"], np.ones((1, 3)))                  # wrong dimension
    idx.delete("delta")
    assert "delta" not in idx.ids and len(idx) == 4


def test_empty_index_returns_nothing():
    assert FlatIndex(4).search(np.ones(4)) == []


def test_ivf_matches_flat_on_easy_queries():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 32)) * 5
    vecs = np.vstack([c + rng.normal(size=(50, 32)) for c in centers])
    ids = [str(i) for i in range(len(vecs))]
    flat, ivf = FlatIndex(32), IVFIndex(32, n_lists=4, n_probe=1)
    flat.add(ids, vecs)
    ivf.add(ids, vecs)
    q = centers[2] + rng.normal(size=32) * 0.1
    flat_top = {r.id for r in flat.search(q, k=10)}
    ivf_top = {r.id for r in ivf.search(q, k=10)}
    assert len(flat_top & ivf_top) / 10 >= 0.8  # recall of the approximate index


def test_ivf_scans_fewer_vectors():
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(200, 16))
    ivf = IVFIndex(16, n_lists=10, n_probe=2)
    ivf.add([str(i) for i in range(200)], vecs)
    assert len(ivf._candidates(ivf.vectors[0])) < 200


TEXT = "First sentence here. Second one is longer than the first! Is this the third? Yes it is."


def test_fixed_size_overlap():
    chunks = fixed_size_chunks("abcdefghij" * 5, size=20, overlap=5)
    assert all(len(c) <= 20 for c in chunks)
    assert chunks[0][-5:] == chunks[1][:5]
    with pytest.raises(ValueError):
        fixed_size_chunks("abc", size=5, overlap=5)


def test_sentence_chunking_keeps_sentences_whole():
    assert len(split_sentences(TEXT)) == 4
    chunks = sentence_chunks(TEXT, max_chars=50, overlap_sentences=0)
    assert all(c.endswith((".", "!", "?")) for c in chunks)
    assert " ".join(chunks) == TEXT


def test_recursive_chunks_respect_limit_and_lose_nothing():
    text = ("Paragraph one. " * 10 + "\n\n" + "Paragraph two. " * 10).strip()
    chunks = recursive_chunks(text, max_chars=60)
    assert all(len(c) <= 60 for c in chunks)
    # no words lost (separators like ". " may be consumed at boundaries)
    assert sum(c.count("Paragraph") for c in chunks) == 20


def test_markdown_sections_keep_heading_path():
    md = "# Guide\nintro\n## Setup\ninstall it\n## Usage\nrun it\n### Advanced\nflags"
    chunks = markdown_section_chunks(md)
    sections = [c.metadata["section"] for c in chunks]
    assert sections == ["Guide", "Guide > Setup", "Guide > Usage", "Guide > Usage > Advanced"]
