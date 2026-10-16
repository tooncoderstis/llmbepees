"""Neuron, lapisan, SwiGLU, dan feed-forward network (Bab 9) — NumPy murni.

Bab 9 membongkar komponen feed-forward (FFN) dari satuan terkecilnya: satu
neuron, lalu satu lapisan (banyak neuron sekaligus), lalu FFN dua lapis.
Juga memperlihatkan SwiGLU — varian FFN bergerbang di Llama/PaLM.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from llmbepees.np.activation import relu, silu
from llmbepees.np.vectors import as_vector

Vector = NDArray[np.float64]
Activation = Callable[[ArrayLike], NDArray[np.float64]]


def pre_activation(x: ArrayLike, w: ArrayLike, b: float) -> float:
    """Pre-aktivasi satu neuron: ``z = x·w + b`` (Bab 9.1)."""
    x = as_vector(x, name="x")
    w = as_vector(w, name="w")
    if x.shape != w.shape:
        raise ValueError(f"bentuk x dan w harus sama: {x.shape} != {w.shape}")
    return float(x @ w + float(b))


def neuron(x: ArrayLike, w: ArrayLike, b: float, activation: Activation = relu) -> float:
    """Satu neuron: ``y = f(x·w + b)`` (Bab 9.1)."""
    z = pre_activation(x, w, b)
    return float(np.asarray(activation(z), dtype=np.float64))


def linear(x: ArrayLike, w: ArrayLike, b: ArrayLike | None = None) -> Vector:
    """Lapisan linear (pre-aktivasi): ``z = xW + b`` (Bab 9.2).

    ``x`` boleh vektor ``(n,)`` atau batch ``(batch, n)``; ``w`` berbentuk
    ``(n, m)``; ``b`` berbentuk ``(m,)`` atau ``None``.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError(f"x harus 1 atau 2 dimensi, dapat ndim={x.ndim}")
    if w.ndim != 2:
        raise ValueError(f"w harus 2 dimensi, dapat ndim={w.ndim}")
    if x.shape[-1] != w.shape[0]:
        raise ValueError(f"dimensi tidak cocok: {x.shape} · {w.shape}")
    z = x @ w
    if b is not None:
        b = np.asarray(b, dtype=np.float64)
        if b.shape != (w.shape[1],):
            raise ValueError(f"b harus berbentuk {(w.shape[1],)}, dapat {b.shape}")
        z = z + b
    return z


def dense(
    x: ArrayLike,
    w: ArrayLike,
    b: ArrayLike | None = None,
    activation: Activation | None = None,
) -> Vector:
    """Lapisan padat: ``y = f(xW + b)``, ``f`` opsional (Bab 9.2)."""
    z = linear(x, w, b)
    return z if activation is None else np.asarray(activation(z), dtype=np.float64)


def layer_parameters(n: int, m: int) -> int:
    """Jumlah parameter lapisan ``n → m``: ``n·m`` bobot ``+ m`` bias (Bab 9.2)."""
    return n * m + m


def collapse_linear_layers(w1: ArrayLike, w2: ArrayLike) -> Vector:
    """Gabung dua lapisan linear tanpa aktivasi: ``W3 = W1·W2`` (Bab 9.3).

    Menunjukkan bahwa menumpuk transformasi affine identik dengan satu lapisan.
    """
    w1 = np.asarray(w1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    if w1.ndim != 2 or w2.ndim != 2:
        raise ValueError("w1 dan w2 harus 2 dimensi")
    if w1.shape[1] != w2.shape[0]:
        raise ValueError(f"dimensi tidak cocok: {w1.shape} · {w2.shape}")
    return w1 @ w2


def ffn(
    x: ArrayLike,
    w1: ArrayLike,
    b1: ArrayLike | None,
    w2: ArrayLike,
    b2: ArrayLike | None,
    activation: Activation = relu,
) -> Vector:
    """Feed-forward network: ``f(xW1 + b1)W2 + b2`` (Bab 9.6).

    ``W1`` ``(d, d_ff)`` menaikkan dimensi, lalu ``W2`` ``(d_ff, d)``
    menurunkannya kembali.
    """
    hidden = dense(x, w1, b1, activation)
    return linear(hidden, w2, b2)


def ffn_parameters(d: int, d_ff: int) -> int:
    """Jumlah parameter FFN: ``2·d·d_ff + d_ff + d`` (Bab 9.6)."""
    return 2 * d * d_ff + d_ff + d


def swiglu(gate: ArrayLike, up: ArrayLike) -> Vector:
    """SwiGLU: ``SiLU(gate) ⊙ up`` (perkalian elemen-per-elemen) (Bab 9.5)."""
    return silu(gate) * np.asarray(up, dtype=np.float64)


def ffn_swiglu(
    x: ArrayLike,
    w_gate: ArrayLike,
    w_up: ArrayLike,
    w_down: ArrayLike,
    b_gate: ArrayLike | None = None,
    b_up: ArrayLike | None = None,
    b_down: ArrayLike | None = None,
) -> Vector:
    """FFN bergerbang: ``SwiGLU(xW_gate, xW_up)·W_down`` (Bab 9.5)."""
    gate = linear(x, w_gate, b_gate)
    up = linear(x, w_up, b_up)
    return linear(swiglu(gate, up), w_down, b_down)


__all__ = [
    "collapse_linear_layers",
    "dense",
    "ffn",
    "ffn_parameters",
    "ffn_swiglu",
    "layer_parameters",
    "linear",
    "neuron",
    "pre_activation",
    "swiglu",
]
