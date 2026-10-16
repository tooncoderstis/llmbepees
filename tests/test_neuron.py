"""Uji neuron, lapisan, SwiGLU, dan FFN (Bab 9) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.activation import relu, silu
from llmbepees.np.neuron import (
    collapse_linear_layers,
    dense,
    ffn,
    ffn_parameters,
    ffn_swiglu,
    layer_parameters,
    linear,
    neuron,
    pre_activation,
    swiglu,
)

TOL = 1e-3

W1_MINI = 0.5 * np.array(
    [
        [1, 0, 1, -1, 0, 1, -1, 0],
        [0, 1, -1, 1, 1, 0, 0, -1],
        [1, 1, 0, 0, -1, 1, 0, 1],
        [-1, 0, 1, 1, 0, -1, 1, 0],
    ],
    dtype=float,
)


def test_pre_activation_example() -> None:
    z = pre_activation([2.0, -1.0, 0.5], [0.5, 1.0, -2.0], 0.3)
    assert z == pytest.approx(-0.7, abs=TOL)


def test_neuron_relu_dies_for_negative_pre_activation() -> None:
    y = neuron([2.0, -1.0, 0.5], [0.5, 1.0, -2.0], 0.3, activation=relu)
    assert y == pytest.approx(0.0)


def test_neuron_custom_activation() -> None:
    y = neuron([2.0, -1.0, 0.5], [0.5, 1.0, -2.0], 0.3, activation=silu)
    assert y == pytest.approx(-0.7 * 0.3318, abs=TOL)


def test_neuron_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        neuron([1.0, 2.0], [1.0], 0.0)


def test_linear_mini_llm_layer() -> None:
    h = [-0.8793, -1.0919, 0.7721, 1.1991]
    z = linear(h, W1_MINI)
    assert z == pytest.approx(
        [-0.6531, -0.1599, 0.7058, 0.4933, -0.9320, -0.6531, 1.0392, 0.9320], abs=TOL
    )


def test_linear_then_relu_mini_llm_layer() -> None:
    h = [-0.8793, -1.0919, 0.7721, 1.1991]
    out = relu(linear(h, W1_MINI))
    assert out == pytest.approx([0.0, 0.0, 0.7058, 0.4933, 0.0, 0.0, 1.0392, 0.9320], abs=TOL)
    assert int(np.count_nonzero(out == 0.0)) == 4  # empat dari delapan neuron mati


def test_linear_with_bias() -> None:
    assert linear([1.0, 1.0], np.eye(2), [0.5, -0.5]) == pytest.approx([1.5, 0.5])


def test_linear_batch() -> None:
    X = [[1.0, 0.0], [0.0, 1.0]]
    Z = linear(X, np.array([[2.0, 3.0], [4.0, 5.0]]))
    assert Z.shape == (2, 2)
    np.testing.assert_allclose(Z, [[2.0, 3.0], [4.0, 5.0]], atol=TOL)


def test_dense_applies_activation() -> None:
    out = dense([-1.0, 2.0], np.eye(2), None, activation=relu)
    assert out == pytest.approx([0.0, 2.0])


def test_layer_parameters() -> None:
    assert layer_parameters(4, 8) == 40
    assert layer_parameters(512, 2048) == 512 * 2048 + 2048


def test_collapse_two_linear_layers() -> None:
    w1 = [[1.0, 2.0], [3.0, 4.0]]
    w2 = [[5.0, 6.0], [7.0, 8.0]]
    np.testing.assert_allclose(collapse_linear_layers(w1, w2), [[19.0, 22.0], [43.0, 50.0]])


def test_two_linear_layers_equal_one() -> None:
    w1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    w2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    w3 = collapse_linear_layers(w1, w2)
    x = np.array([1.0, 1.0])
    assert np.asarray(x @ w1 @ w2).tolist() == [62.0, 72.0]
    assert np.asarray(x @ w3).tolist() == [62.0, 72.0]


def test_nonlinearity_changes_result() -> None:
    w1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    w2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    x = np.array([1.0, -1.0])
    with_relu = np.asarray(relu(x @ w1) @ w2)
    without_relu = np.asarray(x @ w1 @ w2)
    assert with_relu.tolist() == [0.0, 0.0]
    assert without_relu.tolist() == [-24.0, -28.0]


def test_ffn_forward() -> None:
    out = ffn(
        [1.0, 2.0],
        np.eye(2),
        [0.0, 0.0],
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        [0.0, 0.0],
        activation=relu,
    )
    assert out == pytest.approx([3.0, 3.0])


def test_ffn_parameters() -> None:
    assert ffn_parameters(512, 2048) == 2 * 512 * 2048 + 2048 + 512
    assert ffn_parameters(12288, 49152) == 1_208_020_992


def test_d_ff_convention_is_four_d() -> None:
    d = 512
    assert ffn_parameters(d, 4 * d) == 2 * d * (4 * d) + (4 * d) + d


def test_swiglu_example() -> None:
    gate = [2.0, -1.0, 0.5]
    up = [1.0, 3.0, -2.0]
    assert swiglu(gate, up) == pytest.approx([1.7616, -0.8067, -0.6224], abs=TOL)


def test_swiglu_matches_definition() -> None:
    gate = np.array([2.0, -1.0, 0.5])
    up = np.array([1.0, 3.0, -2.0])
    np.testing.assert_allclose(swiglu(gate, up), silu(gate) * up)


def test_ffn_swiglu_shape() -> None:
    x = [0.1, -0.2, 0.3, 0.4]
    w_gate = np.ones((4, 8))
    w_up = np.ones((4, 8))
    w_down = np.ones((8, 4))
    out = ffn_swiglu(x, w_gate, w_up, w_down)
    assert out.shape == (4,)


def test_swiglu_parameter_parity_with_standard_ffn() -> None:
    d, d_ff_swiglu = 4096, 11008
    swiglu_params = 3 * d * d_ff_swiglu
    standard_params = 2 * d * (4 * d)
    assert swiglu_params == 135_266_304
    assert standard_params == 134_217_728
    assert abs(swiglu_params - standard_params) / standard_params < 0.01
