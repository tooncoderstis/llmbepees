"""Uji informasi posisi (Bab 15) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.position import (
    add_position,
    alibi_bias,
    position_interpolation,
    rope_apply,
    rope_rotate,
    rope_theta,
    sinusoidal_encoding,
)

TOL = 1e-3

P_MINI = [
    [0.5, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.5],
    [0.5, 0.5, 0.0, 0.0],
]


def test_add_position_makes_positions_distinguishable() -> None:
    kucing = [1.0, 0.0, 1.0, 0.0]
    assert add_position(kucing, P_MINI[1]) == pytest.approx([1.0, 0.5, 1.0, 0.0])
    assert add_position(kucing, P_MINI[3]) == pytest.approx([1.0, 0.0, 1.0, 0.5])


def test_add_position_batch() -> None:
    X = [P_MINI[0], P_MINI[1]]
    out = add_position(X, P_MINI[:2])
    assert out[0] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert out[1] == pytest.approx([0.0, 1.0, 0.0, 0.0])


def test_sinusoidal_encoding_table() -> None:
    pe = sinusoidal_encoding(5, 4)
    assert pe[0] == pytest.approx([0.0, 1.0, 0.0, 1.0], abs=TOL)
    assert pe[1] == pytest.approx([0.8415, 0.5403, 0.0100, 1.0000], abs=TOL)
    assert pe[2] == pytest.approx([0.9093, -0.4161, 0.0200, 0.9998], abs=TOL)
    assert pe[3] == pytest.approx([0.1411, -0.9900, 0.0300, 0.9996], abs=TOL)
    assert pe[4] == pytest.approx([-0.7568, -0.6536, 0.0400, 0.9992], abs=TOL)


def test_sinusoidal_requires_even_d() -> None:
    with pytest.raises(ValueError):
        sinusoidal_encoding(4, 3)


def test_rope_theta_d4() -> None:
    assert rope_theta(2, 4) == pytest.approx([1.0, 0.01])


def test_rope_rotate_examples() -> None:
    q = [1.0, 0.0, 1.0, 0.0]
    assert rope_rotate(q, 0) == pytest.approx([1.0, 0.0, 1.0, 0.0])
    assert rope_rotate(q, 1) == pytest.approx([0.5403, 0.8415, 1.0000, 0.0100], abs=TOL)
    assert rope_rotate(q, 2) == pytest.approx([-0.4161, 0.9093, 0.9998, 0.0200], abs=TOL)


def test_rope_preserves_length() -> None:
    q = [1.0, 0.0, 1.0, 0.0]
    before = float(np.linalg.norm(q))
    for m in (0, 1, 2, 5):
        assert float(np.linalg.norm(rope_rotate(q, m))) == pytest.approx(before, abs=1e-9)


def test_rope_dot_depends_only_on_relative_position() -> None:
    q = [1.0, 0.0, 1.0, 0.0]
    k = [1.0, 0.0, 1.0, 0.0]
    cases = [
        ((0, 0), 2.0000),
        ((1, 0), 1.5403),
        ((2, 0), 0.5837),
        ((3, 1), 0.5837),
        ((5, 3), 0.5837),
    ]
    for (m, n), expected in cases:
        dot = float(rope_rotate(q, m) @ rope_rotate(k, n))
        assert dot == pytest.approx(expected, abs=TOL)


def test_rope_apply_matches_rotate_per_row() -> None:
    Q = [[1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5]]
    Q_rot, K_rot = rope_apply(Q, Q)
    assert Q_rot[1] == pytest.approx(rope_rotate(Q[1], 1))
    assert K_rot[0] == pytest.approx([1.0, 0.0, 1.0, 0.0])


def test_rope_theta_small_at_d128() -> None:
    theta = rope_theta(64, 128)
    assert theta[0] == pytest.approx(1.0)
    assert theta[63] == pytest.approx(1.2e-4, abs=1e-5)


def test_alibi_bias_penalizes_distance() -> None:
    bias = alibi_bias(11, 0.25)
    assert bias[10, 10] == pytest.approx(0.0)
    assert bias[10, 9] == pytest.approx(-0.25)
    assert bias[10, 6] == pytest.approx(-1.00)
    assert bias[10, 0] == pytest.approx(-2.50)


def test_alibi_applied_to_scores() -> None:
    bias = alibi_bias(11, 0.25)
    raw = 3.0
    assert raw + bias[10, 10] == pytest.approx(3.00)
    assert raw + bias[10, 9] == pytest.approx(2.75)
    assert raw + bias[10, 6] == pytest.approx(2.00)
    assert raw + bias[10, 0] == pytest.approx(0.50)


def test_position_interpolation() -> None:
    assert position_interpolation(8192, scale=4) == pytest.approx(2048)
