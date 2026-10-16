"""Uji attention skala dot-product (Bab 13) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.attention import (
    attention_flops,
    attention_scores,
    attention_weights,
    causal_mask,
    project_qkv,
    scaled_dot_product_attention,
    score_variance,
)

TOL = 1e-3

H = [
    [0.5, 0.0, 0.0, 0.0],
    [1.0, 0.5, 1.0, 0.0],
    [0.0, 1.0, 0.5, 1.0],
    [1.0, 1.0, 0.0, 0.5],
    [0.5, 0.5, 1.0, 1.0],
]
W_Q = [[1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1]]
W_K = [[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]]
W_V = 0.5 * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]])


def test_project_qkv_matches_book() -> None:
    q, k, v = project_qkv(H, W_Q, W_K, W_V)
    assert q[0] == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert q[4] == pytest.approx([2.0, 1.0, 1.0, 2.5])
    assert k[0] == pytest.approx([0.0, 0.5, 0.0, 0.0])
    assert k[2] == pytest.approx([1.5, 1.0, 1.5, 1.0])
    assert v[0] == pytest.approx([0.25, 0.0, 0.25, 0.0])
    assert v[4] == pytest.approx([0.75, 0.75, 0.75, 0.75])


def test_attention_scores_matrix() -> None:
    q, k, _ = project_qkv(H, W_Q, W_K, W_V)
    S = attention_scores(q, k)
    assert S.shape == (5, 5)
    assert S[4] == pytest.approx([0.5, 5.0, 8.0, 5.25, 9.0])
    assert S[0] == pytest.approx([0.25, 1.75, 2.5, 1.75, 3.0])


def test_causal_mask_shape_and_values() -> None:
    m = causal_mask(5)
    assert m.shape == (5, 5)
    assert m[0, 0] == pytest.approx(0.0)
    assert m[0, 1] < -1e8
    assert np.all(m[np.tril_indices(5)] == 0.0)


def test_attention_weights_row_atas() -> None:
    q, k, _ = project_qkv(H, W_Q, W_K, W_V)
    A = attention_weights(q, k, mask=causal_mask(5))
    assert A[4] == pytest.approx([0.0075, 0.0709, 0.3176, 0.0803, 0.5237], abs=TOL)


def test_attention_weights_rows_sum_to_one() -> None:
    q, k, _ = project_qkv(H, W_Q, W_K, W_V)
    A = attention_weights(q, k, mask=causal_mask(5))
    np.testing.assert_allclose(A.sum(axis=1), np.ones(5), atol=1e-9)
    assert A[0] == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])


def test_causal_mask_blocks_future_positions() -> None:
    q, k, _ = project_qkv(H, W_Q, W_K, W_V)
    A = attention_weights(q, k, mask=causal_mask(5))
    future = np.triu_indices(5, k=1)
    np.testing.assert_allclose(A[future], 0.0, atol=1e-12)


def test_scaled_dot_product_attention_output_atas() -> None:
    q, k, v = project_qkv(H, W_Q, W_K, W_V)
    out = scaled_dot_product_attention(q, k, v, mask=causal_mask(5))
    assert out.shape == (5, 4)
    assert out[4] == pytest.approx([0.5851, 0.7243, 0.6491, 0.7884], abs=TOL)
    assert out[0] == pytest.approx([0.25, 0.0, 0.25, 0.0])


def test_score_variance_grows_with_dk() -> None:
    for dk in (4, 64, 512):
        assert score_variance(dk, trials=20_000, seed=0) == pytest.approx(dk, rel=0.1)


def test_attention_flops_breaks_long_context() -> None:
    flops = attention_flops(16_384, 4096, 4096, 4096)
    assert flops["n_squared"] == pytest.approx(4.4e12, rel=0.05)
    assert flops["n_linear"] == pytest.approx(2.2e12, rel=0.05)
    assert flops["n_squared"] > flops["n_linear"]


def test_attention_flops_short_context_dominated_by_projection() -> None:
    flops = attention_flops(1024, 4096, 4096, 4096)
    assert flops["n_linear"] > flops["n_squared"]
