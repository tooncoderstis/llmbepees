"""Uji LayerNorm & RMSNorm (Bab 16) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.layernorm import layernorm, rmsnorm

TOL = 1e-3


def test_layernorm_example() -> None:
    out = layernorm([2.0, 4.0, 4.0, 6.0])
    assert out == pytest.approx([-1.4142, 0.0, 0.0, 1.4142], abs=TOL)


def test_layernorm_output_has_zero_mean_unit_variance() -> None:
    out = layernorm([2.0, 4.0, 4.0, 6.0])
    assert float(np.mean(out)) == pytest.approx(0.0, abs=TOL)
    assert float(np.var(out)) == pytest.approx(1.0, abs=TOL)


def test_layernorm_mini_llm_atas_row() -> None:
    row = [1.2204, 1.1665, 1.6393, 1.7476]
    assert layernorm(row) == pytest.approx([-0.8793, -1.0919, 0.7721, 1.1991], abs=TOL)


def test_layernorm_is_per_token() -> None:
    H = [[1.2204, 1.1665, 1.6393, 1.7476], [0.6250, 0.1250, 0.2500, 0.0000]]
    out = layernorm(H)
    np.testing.assert_allclose(np.mean(out, axis=1), [0.0, 0.0], atol=TOL)
    np.testing.assert_allclose(np.var(out, axis=1), [1.0, 1.0], atol=TOL)


def test_layernorm_with_gamma_and_beta() -> None:
    out = layernorm([2.0, 4.0, 4.0, 6.0], gamma=[1.0] * 4, beta=[1.0] * 4)
    assert out == pytest.approx([-0.4142, 1.0, 1.0, 2.4142], abs=TOL)


def test_rmsnorm_example() -> None:
    out = rmsnorm([2.0, 4.0, 4.0, 6.0])
    assert out == pytest.approx([0.4714, 0.9428, 0.9428, 1.4142], abs=TOL)


def test_rmsnorm_differs_from_layernorm() -> None:
    h = [2.0, 4.0, 4.0, 6.0]
    assert rmsnorm(h) == pytest.approx([0.4714, 0.9428, 0.9428, 1.4142], abs=TOL)
    assert layernorm(h) == pytest.approx([-1.4142, 0.0, 0.0, 1.4142], abs=TOL)


def test_rmsnorm_keeps_positive_constant() -> None:
    assert rmsnorm([1.0, 1.0, 1.0, 1.0]) == pytest.approx([1.0, 1.0, 1.0, 1.0], abs=TOL)
