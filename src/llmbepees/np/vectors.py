"""Operasi vektor (Bab 3) — rujukan kebenaran NumPy murni.

Setiap fungsi adalah transkripsi langsung rumus buku:
penjumlahan, perkalian skalar, dot product, norma, cosine similarity,
dan Hadamard (elemen-per-elemen).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vector = NDArray[np.float64]


def as_vector(x: ArrayLike, *, name: str = "vektor") -> Vector:
    """Ubah ``x`` menjadi vektor float64 1 dimensi; tolak bentuk lain."""
    vec = np.asarray(x, dtype=np.float64)
    if vec.ndim != 1:
        raise ValueError(f"{name} harus 1 dimensi, dapat ndim={vec.ndim}")
    return vec


def add(a: ArrayLike, b: ArrayLike) -> Vector:
    """Penjumlahan vektor element-wise: ``(a + b)_i = a_i + b_i`` (Bab 3.2)."""
    a = as_vector(a, name="a")
    b = as_vector(b, name="b")
    if a.shape != b.shape:
        raise ValueError(f"dimensi harus sama: {a.shape} != {b.shape}")
    return a + b


def scale(c: float, x: ArrayLike) -> Vector:
    """Perkalian skalar: ``(c * x)_i = c * x_i`` (Bab 3.3)."""
    return float(c) * as_vector(x, name="x")


def dot(a: ArrayLike, b: ArrayLike) -> float:
    """Dot product: ``a · b = sum_i a_i * b_i`` — dua vektor jadi satu skalar (Bab 3.4)."""
    a = as_vector(a, name="a")
    b = as_vector(b, name="b")
    if a.shape != b.shape:
        raise ValueError(f"dimensi harus sama: {a.shape} != {b.shape}")
    return float(np.sum(a * b))


def norm_l2(x: ArrayLike) -> float:
    """Norma Euklides (L2): ``||x|| = sqrt(sum_i x_i^2)`` (Bab 3.5)."""
    vec = as_vector(x, name="x")
    return float(np.sqrt(np.sum(vec * vec)))


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine similarity: ``cos(theta) = (a · b) / (||a|| ||b||)`` (Bab 3.6)."""
    a = as_vector(a, name="a")
    b = as_vector(b, name="b")
    denom = norm_l2(a) * norm_l2(b)
    if denom == 0.0:
        raise ValueError("cosine similarity tidak terdefinisi untuk vektor nol")
    return dot(a, b) / denom


def hadamard(a: ArrayLike, b: ArrayLike) -> Vector:
    """Hadamard (elemen-per-elemen): ``(a ⊙ b)_i = a_i * b_i`` (Bab 3.8)."""
    a = as_vector(a, name="a")
    b = as_vector(b, name="b")
    if a.shape != b.shape:
        raise ValueError(f"dimensi harus sama: {a.shape} != {b.shape}")
    return a * b
