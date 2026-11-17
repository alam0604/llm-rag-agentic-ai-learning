import numpy as np
import pytest

from ailearn.attention import (MultiHeadSelfAttention, causal_mask, sample_next_token,
                               scaled_dot_product_attention, sinusoidal_positional_encoding, softmax)


def test_softmax_sums_to_one_and_is_stable():
    x = np.array([1000.0, 1001.0, 1002.0])  # would overflow without max-subtraction
    p = softmax(x)
    assert np.isclose(p.sum(), 1.0)
    assert np.all(np.isfinite(p))
    assert p[2] > p[1] > p[0]


def test_attention_weights_are_a_probability_distribution():
    rng = np.random.default_rng(0)
    q, k, v = rng.normal(size=(3, 5, 8))
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (5, 8)
    assert np.allclose(w.sum(-1), 1.0)


def test_identical_query_key_attends_most_to_itself():
    # one-hot keys: token i's query matches key i strongly
    k = np.eye(4) * 10
    out, w = scaled_dot_product_attention(k, k, np.eye(4))
    assert np.all(np.argmax(w, axis=-1) == np.arange(4))


def test_causal_mask_blocks_future_tokens():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 4))
    _, w = scaled_dot_product_attention(x, x, x, causal_mask(6))
    assert np.allclose(np.triu(w, k=1), 0.0)  # no attention to the future
    assert np.isclose(w[0, 0], 1.0)            # first token can only see itself


def test_multi_head_shapes_and_validation():
    mha = MultiHeadSelfAttention(d_model=16, n_heads=4)
    x = np.random.default_rng(2).normal(size=(7, 16))
    out, w = mha(x)
    assert out.shape == (7, 16)
    assert w.shape == (4, 7, 7)
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(d_model=10, n_heads=3)


def test_positional_encoding_is_unique_per_position():
    pe = sinusoidal_positional_encoding(50, 16)
    assert pe.shape == (50, 16)
    assert len({tuple(np.round(r, 6)) for r in pe}) == 50


def test_greedy_decoding_is_argmax_and_top_k_restricts_choices():
    logits = np.array([0.1, 5.0, 0.2, 4.9])
    assert sample_next_token(logits, temperature=0) == 1
    rng = np.random.default_rng(0)
    picks = {sample_next_token(logits, temperature=1.0, top_k=2, rng=rng) for _ in range(200)}
    assert picks <= {1, 3}


def test_top_p_keeps_smallest_set_covering_p():
    logits = np.log(np.array([0.6, 0.3, 0.05, 0.05]))
    rng = np.random.default_rng(0)
    picks = {sample_next_token(logits, top_p=0.85, rng=rng) for _ in range(300)}
    assert picks <= {0, 1}
