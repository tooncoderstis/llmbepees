"""Uji multi-head attention (Bab 14) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.attention import causal_mask
from llmbepees.np.multihead import (
    concat_heads,
    kv_cache_bytes,
    kv_cache_ratio,
    multi_head_attention,
    split_heads,
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
W_O = 0.5 * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]])


def _run():
    return multi_head_attention(H, W_Q, W_K, W_V, W_O, num_heads=2, mask=causal_mask(5))


def test_split_heads_shapes() -> None:
    heads = split_heads(W_Q, 2)
    assert len(heads) == 2
    assert heads[0].shape == (4, 2)
    np.testing.assert_allclose(heads[0], np.array(W_Q)[:, :2])
    np.testing.assert_allclose(heads[1], np.array(W_Q)[:, 2:])


def test_split_heads_rejects_uneven() -> None:
    with pytest.raises(ValueError):
        split_heads(np.ones((4, 5)), 2)


def test_concat_heads_shape() -> None:
    a = np.ones((5, 2))
    b = np.zeros((5, 2))
    out = concat_heads([a, b])
    assert out.shape == (5, 4)
    np.testing.assert_allclose(out[:, :2], 1.0)
    np.testing.assert_allclose(out[:, 2:], 0.0)


def test_head1_weights_row_di() -> None:
    _, _, weights = _run()
    assert weights[0][3] == pytest.approx([0.0121, 0.3481, 0.3481, 0.2917, 0.0], abs=TOL)


def test_head2_weights_row_di() -> None:
    _, _, weights = _run()
    assert weights[1][3] == pytest.approx([0.0306, 0.1260, 0.7379, 0.1056, 0.0], abs=TOL)


def test_two_heads_produce_different_patterns() -> None:
    _, _, weights = _run()
    assert not np.allclose(weights[0][3], weights[1][3], atol=0.1)


def test_head_outputs_atas() -> None:
    _, heads, _ = _run()
    assert heads[0][4] == pytest.approx([0.6389, 0.6933], abs=TOL)
    assert heads[1][4] == pytest.approx([0.6397, 0.8019], abs=TOL)


def test_final_output_atas() -> None:
    out, _, _ = _run()
    assert out.shape == (5, 4)
    assert out[4] == pytest.approx([0.7204, 0.6665, 0.6393, 0.7476], abs=TOL)


def test_output_mixes_both_heads_via_w_o() -> None:
    out, heads, _ = _run()
    mixed = 0.5 * heads[0][4, 0] + 0.5 * heads[1][4, 1]
    assert out[4, 0] == pytest.approx(mixed, abs=TOL)


def test_multi_head_keeps_parameter_count() -> None:
    one_head = np.asarray(W_Q).shape[1]
    two_head = sum(h.shape[1] for h in split_heads(W_Q, 2))
    assert two_head == one_head


def test_kv_cache_size() -> None:
    total = kv_cache_bytes(32, 32, 2048, 128, bytes_per_value=2)
    assert total == pytest.approx(2 * 32 * 32 * 2048 * 128 * 2)
    assert total / (1024**3) == pytest.approx(1.0, abs=0.01)


def test_kv_cache_gqa_and_mqa_reduce_size() -> None:
    mha = kv_cache_bytes(32, 32, 2048, 128, bytes_per_value=2)
    gqa = kv_cache_bytes(32, 32, 2048, 128, bytes_per_value=2, num_kv_heads=8)
    mqa = kv_cache_bytes(32, 32, 2048, 128, bytes_per_value=2, num_kv_heads=1)
    assert kv_cache_ratio(32, 8) == pytest.approx(0.25)
    assert kv_cache_ratio(32, 1) == pytest.approx(0.03125)
    assert gqa == pytest.approx(0.25 * mha)
    assert mqa == pytest.approx(mha / 32)
