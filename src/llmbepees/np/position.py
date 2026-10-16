"""Informasi posisi (Bab 15) — rujukan kebenaran NumPy murni.

Attention membandingkan **isi**, bukan urutan: tanpa bantuan, "kucing menggigit
anjing" dan "anjing menggigit kucing" diproses identik. Modul ini menyediakan
empat cara menyuntikkan posisi: embedding terlatih (dijumlahkan), sinusoidal
(dihitung), RoPE (memutar query/key), dan ALiBi (penalti ke skor).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]

BASE = 10000.0


def add_position(X: ArrayLike, P: ArrayLike) -> Matrix:
    """Positional embedding terlatih: ``H(0) = X + P`` (Bab 15.3).

    ``X`` = embedding token, ``P`` = tabel embedding posisi (satu baris per posisi).
    """
    x = np.asarray(X, dtype=np.float64)
    p = np.asarray(P, dtype=np.float64)
    if x.shape[-1] != p.shape[-1]:
        raise ValueError(f"dimensi tidak cocok: {x.shape} + {p.shape}")
    return x + p


def sinusoidal_encoding(max_len: int, d: int, *, base: float = BASE) -> Matrix:
    """Sinusoidal positional encoding ``max_len × d`` (Bab 15.4).

    ``PE(pos, 2i) = sin(pos / base^{2i/d})``, ``PE(pos, 2i+1) = cos(...)``.
    """
    if d % 2 != 0:
        raise ValueError("d harus genap")
    pos = np.arange(max_len, dtype=np.float64)[:, None]
    i = np.arange(d // 2, dtype=np.float64)[None, :]
    angles = pos / (float(base) ** (2.0 * i / d))
    pe = np.zeros((max_len, d), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


def rope_theta(num_pairs: int, d: int, *, base: float = BASE) -> Vector:
    """Kecepatan putar RoPE per pasangan: ``θ_i = base^{−2i/d}`` (Bab 15.5)."""
    i = np.arange(num_pairs, dtype=np.float64)
    return float(base) ** (-2.0 * i / d)


def rope_rotate(q: ArrayLike, position: int, *, base: float = BASE) -> Vector:
    """Putar vektor ``q`` sebesar ``position·θ_i`` per pasangan komponen (Bab 15.5).

    Rotasi 2×2 menjaga panjang vektor; diterapkan pada query/key di tiap lapisan.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError(f"q harus 1 dimensi, dapat ndim={q.ndim}")
    if q.shape[0] % 2 != 0:
        raise ValueError("panjang q harus genap")
    pairs = q.reshape(-1, 2)
    theta = rope_theta(pairs.shape[0], q.shape[0], base=base)
    angle = float(position) * theta
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    out = np.empty_like(pairs)
    out[:, 0] = pairs[:, 0] * cos_a - pairs[:, 1] * sin_a
    out[:, 1] = pairs[:, 0] * sin_a + pairs[:, 1] * cos_a
    return out.reshape(-1)


def rope_apply(Q: ArrayLike, K: ArrayLike, *, base: float = BASE) -> tuple[Matrix, Matrix]:
    """Putar query & key dengan RoPE memakai indeks baris sebagai posisi (Bab 15.5)."""
    q = np.asarray(Q, dtype=np.float64)
    k = np.asarray(K, dtype=np.float64)
    positions = range(q.shape[0])
    q_rot = np.stack([rope_rotate(row, m, base=base) for row, m in zip(q, positions, strict=True)])
    k_rot = np.stack([rope_rotate(row, m, base=base) for row, m in zip(k, positions, strict=True)])
    return q_rot, k_rot


def alibi_bias(n: int, m_h: float) -> Matrix:
    """Bias ALiBi ``n×n``: ``−m_h·(i − j)`` (Bab 15.6).

    Ditambahkan ke skor attention (setelah dibagi ``√d_k``); makin jauh, makin
    dihukum, dengan penalti linear terhadap jarak.
    """
    i = np.arange(n, dtype=np.float64)[:, None]
    j = np.arange(n, dtype=np.float64)[None, :]
    return -float(m_h) * (i - j)


def position_interpolation(position: float, *, scale: float) -> float:
    """Position interpolation: perlakukan posisi panjang sebagai posisi pendek (Bab 15.5).

    Mis. ``position_interpolation(8192, scale=4) = 2048``.
    """
    if scale <= 0:
        raise ValueError("scale harus positif")
    return float(position) / float(scale)


__all__ = [
    "add_position",
    "alibi_bias",
    "position_interpolation",
    "rope_apply",
    "rope_rotate",
    "rope_theta",
    "sinusoidal_encoding",
]
