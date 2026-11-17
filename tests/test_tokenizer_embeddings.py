import numpy as np
import pytest

from ailearn.embeddings import (HashingEmbedder, TfidfEmbedder, cosine_similarity, dot_product,
                                euclidean_distance, tokenize)
from ailearn.tokenizer import BPETokenizer, estimate_tokens

CORPUS = "low lower lowest newer newest wider widest " * 20


def test_bpe_roundtrip_and_compression():
    tok = BPETokenizer().train(CORPUS, vocab_size=300)
    text = "lowest newest"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert len(ids) < len(text.encode())  # merges compress the byte sequence


def test_bpe_handles_unicode_and_unseen_text():
    tok = BPETokenizer().train(CORPUS, vocab_size=280)
    for s in ["नमस्ते", "emoji 🚀", "SELECT * FROM t"]:
        assert tok.decode(tok.encode(s)) == s


def test_bpe_learns_frequent_subwords():
    tok = BPETokenizer().train(CORPUS, vocab_size=320)
    assert any(len(t) >= 3 for t in tok.tokens("lowest"))


def test_bpe_rejects_tiny_vocab():
    with pytest.raises(ValueError):
        BPETokenizer().train("abc", 100)


def test_estimate_tokens():
    assert estimate_tokens("a" * 400) == 100


def test_similarity_metrics():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert np.isclose(cosine_similarity(a, a)[0, 0], 1.0)
    assert np.isclose(cosine_similarity(a, b)[0, 0], 0.0)
    assert np.isclose(euclidean_distance(a, b)[0, 0], np.sqrt(2))
    assert np.isclose(dot_product(a, a)[0, 0], 1.0)
    # cosine ignores magnitude, dot product doesn't
    assert np.isclose(cosine_similarity(a, 5 * a)[0, 0], 1.0)
    assert dot_product(a, 5 * a)[0, 0] == 5.0


@pytest.mark.parametrize("embedder_cls", [TfidfEmbedder, HashingEmbedder])
def test_related_texts_are_closer(embedder_cls, docs):
    emb = embedder_cls()
    if isinstance(emb, TfidfEmbedder):
        emb.fit(list(docs.values()))
    v = emb.embed(["delta lake time travel versions", docs["delta"], docs["adf"]])
    sims = cosine_similarity(v[:1], v[1:])[0]
    assert sims[0] > sims[1]


def test_embeddings_are_unit_length(docs):
    v = HashingEmbedder(dim=64).embed(list(docs.values()))
    assert v.shape == (5, 64)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_tokenize_removes_stopwords():
    assert tokenize("What is the Medallion architecture?") == ["medallion", "architecture"]


def test_tfidf_requires_fit():
    with pytest.raises(RuntimeError):
        TfidfEmbedder().embed(["x"])
