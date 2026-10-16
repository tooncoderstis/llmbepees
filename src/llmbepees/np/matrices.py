"""Operasi matriks (Bab 4) — rujukan kebenaran NumPy murni.

Transkripsi langsung rumus buku: transpos, penjumlahan, perkalian matriks
(elemen hasil = dot product baris × kolom), vektor-dikali-matriks, dan biaya FLOP.

Konvensi buku: satu baris = satu token, satu kolom = satu dimensi fitur.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]


def as_matrix(x: ArrayLike, *, name: str = "matriks") -> Matrix:
    """Ubah ``x`` menjadi matriks float64 2 dimensi; tolak bentuk lain."""
    mat = np.asarray(x, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"{name} harus 2 dimensi, dapat ndim={mat.ndim}")
    return mat


def as_vector(x: ArrayLike, *, name: str = "vektor") -> Vector:
    """Ubah ``x`` menjadi vektor float64 1 dimensi; tolak bentuk lain."""
    vec = np.asarray(x, dtype=np.float64)
    if vec.ndim != 1:
        raise ValueError(f"{name} harus 1 dimensi, dapat ndim={vec.ndim}")
    return vec


def transpose(a: ArrayLike) -> Matrix:
    """Transpos: ``(Aᵀ)_ij = A_ji``; tukar baris dan kolom (Bab 4.2)."""
    return as_matrix(a, name="a").T


def matadd(a: ArrayLike, b: ArrayLike) -> Matrix:
    """Penjumlahan matriks element-wise; ukuran harus identik (Bab 4.3)."""
    a = as_matrix(a, name="a")
    b = as_matrix(b, name="b")
    if a.shape != b.shape:
        raise ValueError(f"ukuran harus sama: {a.shape} != {b.shape}")
    return a + b


def matmul(a: ArrayLike, b: ArrayLike) -> Matrix:
    """Perkalian matriks: ``C_ij = ∑_p A_ip B_pj`` (Bab 4.4).

    Dimensi tengah harus cocok: ``(m × k) · (k × n) = (m × n)``.
    """
    a = as_matrix(a, name="a")
    b = as_matrix(b, name="b")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"dimensi tengah harus cocok: {a.shape} · {b.shape}")
    c = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            c[i, j] = np.sum(a[i, :] * b[:, j])
    return c


def matvec(x: ArrayLike, w: ArrayLike) -> Vector:
    """Vektor baris dikali matriks: ``y_j = ∑_i x_i W_ij`` (Bab 4.6).

    ``x`` berukuran ``1 × d``, ``w`` berukuran ``d × d'``, hasil ``1 × d'``.
    """
    x = as_vector(x, name="x")
    w = as_matrix(w, name="w")
    d, n = w.shape
    if x.shape[0] != d:
        raise ValueError(f"dimensi tidak cocok: {x.shape} · {w.shape}")
    y = np.zeros(n, dtype=np.float64)
    for j in range(n):
        y[j] = np.sum(x * w[:, j])
    return y


def identity(n: int) -> Matrix:
    """Matriks identitas ``n × n`` (elemen 1 di diagonal, 0 di tempat lain)."""
    if n <= 0:
        raise ValueError("n harus positif")
    return np.eye(n, dtype=np.float64)


def flops_matmul(m: int, n: int, k: int) -> int:
    """Perkiraan biaya perkalian ``(m × k)·(k × n)``: ``≈ 2·m·n·k`` FLOP (Bab 4.8)."""
    return 2 * m * n * k
