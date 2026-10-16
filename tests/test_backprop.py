"""Uji backpropagation (Bab 11) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.backprop import (
    activation_memory,
    add_backward,
    factor_power,
    forward_mlp,
    hadamard_backward,
    matmul_backward,
    relu_backward,
    residual_backward,
    residual_gradient,
    training_flops,
    training_time_seconds,
    two_layer_mlp_backward,
)
from llmbepees.np.calculus import numerical_gradient

TOL = 1e-3


def test_forward_mlp_example() -> None:
    z1, h, y = forward_mlp(1.0, 0.5, 0.0, 2.0, 1.0)
    assert (z1, h, y) == pytest.approx((0.5, 0.5, 2.0))


def test_two_layer_mlp_backward_example() -> None:
    grads = two_layer_mlp_backward(1.0, 0.5, 0.0, 2.0, 1.0, 3.0)
    assert grads["loss"] == pytest.approx(1.0)
    assert grads["dL_dy"] == pytest.approx(-2.0)
    assert grads["dL_dw2"] == pytest.approx(-1.0)
    assert grads["dL_db2"] == pytest.approx(-2.0)
    assert grads["dL_dh"] == pytest.approx(-4.0)
    assert grads["dL_dz1"] == pytest.approx(-4.0)
    assert grads["dL_dw1"] == pytest.approx(-4.0)
    assert grads["dL_db1"] == pytest.approx(-4.0)


def test_parameter_update_reduces_loss() -> None:
    g = two_layer_mlp_backward(1.0, 0.5, 0.0, 2.0, 1.0, 3.0)
    lr = 0.1
    w1 = 0.5 - lr * g["dL_dw1"]
    b1 = 0.0 - lr * g["dL_db1"]
    w2 = 2.0 - lr * g["dL_dw2"]
    b2 = 1.0 - lr * g["dL_db2"]
    assert (w1, b1, w2, b2) == pytest.approx((0.9, 0.4, 2.1, 1.2))

    _, _, y = forward_mlp(1.0, w1, b1, w2, b2)
    new_loss = (y - 3.0) ** 2
    assert y == pytest.approx(3.93, abs=TOL)
    assert new_loss == pytest.approx(0.8649, abs=TOL)
    assert new_loss < g["loss"]


def test_add_backward_passes_gradient_through() -> None:
    assert add_backward([0.5, -1.0]) == pytest.approx([0.5, -1.0])


def test_relu_backward_masks_negative_inputs() -> None:
    assert relu_backward([0.5, -0.5, 2.0], [1.0, 1.0, 3.0]) == pytest.approx([1.0, 0.0, 3.0])


def test_hadamard_backward() -> None:
    grad_a, grad_b = hadamard_backward([2.0, -1.0], [3.0, 4.0], [1.0, 0.5])
    assert grad_a == pytest.approx([3.0, 2.0])
    assert grad_b == pytest.approx([2.0, -0.5])


def test_matmul_backward_vector_example() -> None:
    grad_w, grad_x = matmul_backward([2.0, 3.0], np.eye(2), [0.5, -1.0])
    np.testing.assert_allclose(grad_w, [[1.0, -2.0], [1.5, -3.0]], atol=TOL)
    assert grad_x == pytest.approx([0.5, -1.0])


def test_matmul_backward_batch() -> None:
    X = np.eye(2)
    W = np.array([[2.0, 3.0], [4.0, 5.0]])
    G = np.ones((2, 2))
    grad_w, grad_x = matmul_backward(X, W, G)
    np.testing.assert_allclose(grad_w, G, atol=TOL)
    np.testing.assert_allclose(grad_x, [[5.0, 9.0], [5.0, 9.0]], atol=TOL)


def test_matmul_backward_batch_sums_token_contributions() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    G = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    grad_w, _ = matmul_backward(X, np.ones((2, 2)), G)
    expected = sum(np.outer(X[i], G[i]) for i in range(3))
    np.testing.assert_allclose(grad_w, expected, atol=TOL)


def test_residual_backward_is_identity_path() -> None:
    assert residual_backward([1.0, 2.0, 3.0]) == pytest.approx([1.0, 2.0, 3.0])


def test_residual_gradient_matches_numerical() -> None:
    W = np.array([[0.5, -0.2], [0.3, 0.4]])
    grad_out = np.array([1.0, 1.0])

    def loss(x: np.ndarray) -> float:
        return float(np.sum(x + W @ x))

    grad = residual_gradient(grad_out, W)
    np.testing.assert_allclose(grad, numerical_gradient(loss, [0.0, 0.0]), atol=1e-6)


def test_factor_power_table() -> None:
    cases = [
        (0.8, 12, 6.9e-2),
        (0.8, 48, 2.2e-5),
        (0.8, 96, 5.0e-10),
        (0.95, 12, 0.540),
        (0.95, 48, 0.085),
        (0.95, 96, 0.0073),
        (1.05, 12, 1.80),
        (1.05, 48, 10.4),
        (1.05, 96, 108.2),
        (1.2, 12, 8.9),
        (1.2, 48, 6320.0),
        (1.2, 96, 4.0e7),
    ]
    for factor, depth, expected in cases:
        assert factor_power(factor, depth) == pytest.approx(expected, rel=0.05)


def test_residual_identity_path_keeps_gradient() -> None:
    assert factor_power(1.0, 96) == pytest.approx(1.0)


def test_training_flops() -> None:
    assert training_flops(7e9, 2e12) == pytest.approx(8.4e22)
    seconds = training_time_seconds(8.4e22, 1.5e14)
    assert seconds == pytest.approx(5.6e8, rel=0.01)


def test_activation_memory() -> None:
    per_sequence = activation_memory(32, 2048, 4096, k=12, bytes_per_value=2)
    assert per_sequence == pytest.approx(6.4e9, rel=0.01)
    assert 8 * per_sequence == pytest.approx(51e9, rel=0.02)
