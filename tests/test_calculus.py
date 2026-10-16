"""Uji turunan, gradien, dan aturan rantai (Bab 7) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import math

import numpy as np
import pytest

from llmbepees.np.calculus import (
    central_difference,
    gradient_descent_step,
    linear_backward,
    numerical_derivative,
    numerical_gradient,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    tanh_derivative,
)

TOL = 1e-3


def test_derivative_x_squared_at_3() -> None:
    assert central_difference(lambda x: x**2, 3.0) == pytest.approx(6.0, abs=TOL)


def test_forward_difference() -> None:
    assert numerical_derivative(lambda x: x**2, 3.0, method="forward") == pytest.approx(
        6.0, abs=1e-2
    )


def test_numerical_derivative_unknown_method() -> None:
    with pytest.raises(ValueError):
        numerical_derivative(lambda x: x, 1.0, method="backward")


def test_chain_rule_composed_function() -> None:
    def f(x: float) -> float:
        return (3.0 * x + 1.0) ** 2

    assert central_difference(f, 2.0) == pytest.approx(42.0, abs=TOL)


def test_partial_derivative_of_two_variable_function() -> None:
    def f(v: np.ndarray) -> float:
        return v[0] ** 2 * v[1] + 3.0 * v[1]

    grad = numerical_gradient(f, [2.0, 5.0])
    assert grad == pytest.approx([20.0, 7.0], abs=TOL)


def test_gradient_points_of_minimum() -> None:
    def f(v: np.ndarray) -> float:
        return (v[0] - 3.0) ** 2

    assert numerical_gradient(f, [0.0])[0] == pytest.approx(-6.0, abs=TOL)


def test_gradient_descent_step_rule() -> None:
    assert gradient_descent_step([0.0], [-6.0], lr=0.1) == pytest.approx([0.6])


def test_gradient_descent_converges_to_minimum() -> None:
    w = np.array([0.0])
    for _ in range(50):
        grad = numerical_gradient(lambda v: (v[0] - 3.0) ** 2, w)
        w = gradient_descent_step(w, grad, lr=0.1)
    assert w[0] == pytest.approx(3.0, abs=1e-3)


def test_gradient_descent_diverges_with_large_lr() -> None:
    w = np.array([0.0])
    for _ in range(4):
        grad = np.array([2.0 * (w[0] - 3.0)])
        w = gradient_descent_step(w, grad, lr=1.1)
    assert abs(w[0]) > 3.0


def test_relu_and_derivative() -> None:
    assert relu([2.0, -1.0, 0.5, -3.0]) == pytest.approx([2.0, 0.0, 0.5, 0.0])
    assert relu_derivative([2.0, -1.0, 0.5, -3.0]) == pytest.approx([1.0, 0.0, 1.0, 0.0])


def test_sigmoid_derivative_maximum() -> None:
    assert sigmoid(0.0) == pytest.approx(0.5, abs=TOL)
    assert sigmoid_derivative(0.0) == pytest.approx(0.25, abs=TOL)


def test_sigmoid_derivative_saturates() -> None:
    assert sigmoid_derivative(5.0) == pytest.approx(0.0067, abs=TOL)


def test_tanh_derivative_at_zero() -> None:
    assert tanh_derivative(0.0) == pytest.approx(1.0, abs=TOL)


def test_linear_backward_grad_w() -> None:
    grad_w, _ = linear_backward([2.0, 3.0], [[1.0, 0.0], [0.0, 1.0]], [0.5, -1.0])
    np.testing.assert_allclose(grad_w, [[1.0, -2.0], [1.5, -3.0]], atol=TOL)


def test_numeric_gradient_of_softmax_cross_entropy_is_p_minus_y() -> None:
    z = np.array([2.0, 1.0, 0.1])
    y = np.array([1.0, 0.0, 0.0])

    def loss(logits: np.ndarray) -> float:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        p = exp / np.sum(exp)
        return float(-np.sum(y * np.log(p)))

    grad = numerical_gradient(loss, z)
    p = np.exp(z - np.max(z))
    p = p / np.sum(p)
    np.testing.assert_allclose(grad, p - y, atol=1e-4)


def test_gradient_descent_step_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        gradient_descent_step([1.0, 2.0], [1.0], lr=0.1)


def test_derivative_of_exp_and_log() -> None:
    assert central_difference(math.exp, 1.0) == pytest.approx(math.e, abs=TOL)
    assert central_difference(math.log, 2.0) == pytest.approx(0.5, abs=TOL)
