"""Turunan, gradien, dan aturan rantai (Bab 7) — rujukan kebenaran NumPy murni.

Turunan numerik (beda hingga), gradien, langkah gradient descent, turunan
fungsi aktivasi yang dipakai LLM, dan aturan turunan ``y = xW``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vector = NDArray[np.float64]
ScalarFn = Callable[[float], float]
VectorFn = Callable[[Vector], float]


def as_vector(x: ArrayLike, *, name: str = "x") -> Vector:
    """Ubah ``x`` menjadi vektor float64 1 dimensi; tolak bentuk lain."""
    vec = np.asarray(x, dtype=np.float64)
    if vec.ndim != 1:
        raise ValueError(f"{name} harus 1 dimensi, dapat ndim={vec.ndim}")
    return vec


def central_difference(f: ScalarFn, x: float, *, h: float = 1e-5) -> float:
    """Beda hingga terpusat: ``(f(x+h) − f(x−h)) / (2h)`` (Bab 7.9)."""
    return float((f(x + h) - f(x - h)) / (2.0 * h))


def numerical_derivative(
    f: ScalarFn,
    x: float,
    *,
    h: float = 1e-5,
    method: str = "central",
) -> float:
    """Turunan numerik ``f'(x)`` (Bab 7.1).

    ``method="central"`` → beda terpusat; ``method="forward"`` → ``(f(x+h) − f(x))/h``.
    """
    if method == "central":
        return central_difference(f, x, h=h)
    if method == "forward":
        return float((f(x + h) - f(x)) / h)
    raise ValueError(f"method tidak dikenal: {method!r}")


def numerical_gradient(f: VectorFn, x: ArrayLike, *, h: float = 1e-5) -> Vector:
    """Gradien numerik lewat beda terpusat per komponen (Bab 7.5)."""
    x = as_vector(x, name="x")
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        step = np.zeros_like(x)
        step[i] = h
        grad[i] = (f(x + step) - f(x - step)) / (2.0 * h)
    return grad


def gradient_descent_step(params: ArrayLike, grad: ArrayLike, *, lr: float) -> Vector:
    """Aturan pembaruan: ``θ ← θ − η·∇L`` (Bab 7.3, 7.5)."""
    params = as_vector(params, name="params")
    grad = as_vector(grad, name="grad")
    if params.shape != grad.shape:
        raise ValueError(f"bentuk harus sama: {params.shape} != {grad.shape}")
    return params - float(lr) * grad


def relu(x: ArrayLike) -> Vector:
    """ReLU: ``max(0, x)`` element-wise."""
    return np.maximum(0.0, np.asarray(x, dtype=np.float64))


def relu_derivative(x: ArrayLike) -> Vector:
    """Turunan ReLU: 1 jika ``x > 0``, 0 jika ``x < 0`` (Bab 7.7)."""
    return (np.asarray(x, dtype=np.float64) > 0.0).astype(np.float64)


def sigmoid(x: ArrayLike) -> Vector:
    """Sigmoid: ``σ(x) = 1 / (1 + e^{−x})`` (Bab 7.7)."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def sigmoid_derivative(x: ArrayLike) -> Vector:
    """Turunan sigmoid: ``σ(x)(1 − σ(x))`` (Bab 7.7)."""
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_derivative(x: ArrayLike) -> Vector:
    """Turunan tanh: ``1 − tanh²(x)`` (Bab 7.7)."""
    t = np.tanh(np.asarray(x, dtype=np.float64))
    return 1.0 - t * t


def linear_backward(x: ArrayLike, w: ArrayLike, grad_y: ArrayLike) -> tuple[Vector, Vector]:
    """Gradien aturan ``y = xW`` (Bab 7.8).

    Diberikan ``x`` (``1×d``), ``w`` (``d×d'``), dan ``∂L/∂y`` (``1×d'``):
    ``∂L/∂W = xᵀ (∂L/∂y)`` (``d×d'``) dan ``∂L/∂x = (∂L/∂y) Wᵀ`` (``1×d``).
    Mengembalikan ``(grad_w, grad_x)``.
    """
    x = as_vector(x, name="x")
    grad_y = as_vector(grad_y, name="grad_y")
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"w harus 2 dimensi, dapat ndim={w.ndim}")
    if w.shape[0] != x.shape[0]:
        raise ValueError(f"dimensi tidak cocok: {x.shape} · {w.shape}")
    grad_w = np.outer(x, grad_y)
    grad_x = grad_y @ w.T
    return grad_w, grad_x
