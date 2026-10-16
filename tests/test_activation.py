"""Uji katalog fungsi aktivasi (Bab 9) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.activation import (
    ACTIVATIONS,
    gelu,
    gelu_approx,
    gelu_derivative,
    get_activation,
    relu,
    sigmoid,
    silu,
    silu_derivative,
    tanh,
)
from llmbepees.np.calculus import numerical_derivative

TOL = 1e-3

XS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def test_relu_table() -> None:
    assert relu(XS) == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 2.0])


def test_gelu_approx_table() -> None:
    assert gelu_approx(XS) == pytest.approx(
        [-0.0454, -0.1588, -0.1543, 0.0, 0.3457, 0.8412, 1.9546], abs=TOL
    )


def test_gelu_exact_table() -> None:
    assert gelu(XS) == pytest.approx(
        [-0.0455, -0.1587, -0.1543, 0.0, 0.3457, 0.8413, 1.9545], abs=TOL
    )


def test_gelu_approx_close_to_exact() -> None:
    assert gelu(XS) == pytest.approx(np.asarray(gelu_approx(XS)), abs=2e-4)


def test_sigmoid_table() -> None:
    assert sigmoid(XS) == pytest.approx(
        [0.1192, 0.2689, 0.3775, 0.5, 0.6225, 0.7311, 0.8808], abs=TOL
    )


def test_tanh_table() -> None:
    assert tanh(XS) == pytest.approx(
        [-0.9640, -0.7616, -0.4621, 0.0, 0.4621, 0.7616, 0.9640], abs=TOL
    )


def test_silu_table() -> None:
    assert silu(XS) == pytest.approx(
        [-0.2384, -0.2689, -0.1888, 0.0, 0.3112, 0.7311, 1.7616], abs=TOL
    )


def test_silu_equals_x_times_sigmoid() -> None:
    assert silu(XS) == pytest.approx(np.asarray(XS) * np.asarray(sigmoid(XS)))


def test_gelu_negative_for_small_negative_input() -> None:
    assert gelu(-1.0) < 0.0
    assert gelu_approx(-1.0) < 0.0
    assert relu(-1.0) == pytest.approx(0.0)


def test_gelu_derivative_matches_numerical() -> None:
    for x in (-1.3, -0.4, 0.0, 0.7, 1.9):
        assert gelu_derivative(x) == pytest.approx(numerical_derivative(gelu, x), abs=1e-4)


def test_silu_derivative_matches_numerical() -> None:
    for x in (-1.3, -0.4, 0.0, 0.7, 1.9):
        assert silu_derivative(x) == pytest.approx(numerical_derivative(silu, x), abs=1e-4)


def test_get_activation_registry() -> None:
    assert get_activation("relu") is relu
    assert get_activation("gelu") is gelu
    assert set(ACTIVATIONS) == {"relu", "sigmoid", "tanh", "gelu", "gelu_approx", "silu"}


def test_get_activation_unknown() -> None:
    with pytest.raises(ValueError):
        get_activation("elu")
