"""Embedding (Bab 8) — rujukan kebenaran NumPy murni.

One-hot encoding, tabel embedding (pencarian baris), kesetaraan ``X = OE``,
dan weight tying (``logit = h Eᵀ``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]
IntVector = NDArray[np.int64]


def _as_matrix(x: ArrayLike, *, name: str = "x") -> Matrix:
    mat = np.asarray(x, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"{name} harus 2 dimensi, dapat ndim={mat.ndim}")
    return mat


def _as_vector(x: ArrayLike, *, name: str = "x") -> Vector:
    vec = np.asarray(x, dtype=np.float64)
    if vec.ndim != 1:
        raise ValueError(f"{name} harus 1 dimensi, dapat ndim={vec.ndim}")
    return vec


def _as_ids(ids: ArrayLike, *, name: str = "ids") -> IntVector:
    arr = np.asarray(ids)
    if arr.ndim != 1:
        raise ValueError(f"{name} harus 1 dimensi, dapat ndim={arr.ndim}")
    return arr.astype(np.int64)


def one_hot(token_id: int, vocab_size: int) -> Vector:
    """One-hot: vektor panjang ``vocab_size`` bernilai 1 di ``token_id`` (Bab 8.3)."""
    if vocab_size <= 0:
        raise ValueError("vocab_size harus positif")
    if not 0 <= token_id < vocab_size:
        raise ValueError(f"token_id {token_id} di luar rentang [0, {vocab_size})")
    vec = np.zeros(vocab_size, dtype=np.float64)
    vec[token_id] = 1.0
    return vec


def one_hot_matrix(ids: ArrayLike, vocab_size: int) -> Matrix:
    """One-hot banyak token: matriks ``O ∈ ℝ^{n×V}`` (Bab 8.7)."""
    ids = _as_ids(ids)
    if vocab_size <= 0:
        raise ValueError("vocab_size harus positif")
    if np.any((ids < 0) | (ids >= vocab_size)):
        raise ValueError(f"semua id harus di [0, {vocab_size})")
    mat = np.zeros((ids.shape[0], vocab_size), dtype=np.float64)
    mat[np.arange(ids.shape[0]), ids] = 1.0
    return mat


def embedding_lookup(table: ArrayLike, ids: ArrayLike) -> Matrix:
    """Ambil baris embedding untuk tiap id: ``X[i] = E[ids[i]]`` (Bab 8.4).

    Inilah implementasi nyata: biaya ``O(d)`` per token, bukan ``O(V·d)``.
    """
    table = _as_matrix(table, name="table")
    ids = _as_ids(ids)
    if np.any((ids < 0) | (ids >= table.shape[0])):
        raise ValueError(f"semua id harus di [0, {table.shape[0]})")
    return table[ids]


def embed_one_hot(table: ArrayLike, one_hots: ArrayLike) -> Matrix:
    """Embedding lewat perkalian matriks ``X = OE`` (Bab 8.4/8.7).

    Hasilnya sama dengan :func:`embedding_lookup`; dipakai untuk menunjukkan
    kesetaraan rumus ``e_i = o_i E``.
    """
    table = _as_matrix(table, name="table")
    one_hots = _as_matrix(one_hots, name="one_hots")
    if one_hots.shape[1] != table.shape[0]:
        raise ValueError(f"dimensi tidak cocok: {one_hots.shape} · {table.shape}")
    return one_hots @ table


def tied_logits(table: ArrayLike, h: ArrayLike) -> Vector:
    """Weight tying: logit ``= h Eᵀ`` untuk setiap token (Bab 8.6)."""
    table = _as_matrix(table, name="table")
    h = _as_vector(h, name="h")
    if table.shape[1] != h.shape[0]:
        raise ValueError(f"dimensi tidak cocok: h {h.shape} vs E {table.shape}")
    return table @ h


def embedding_parameters(vocab_size: int, d_model: int) -> int:
    """Jumlah parameter matriks embedding: ``V × d`` (Bab 8.4)."""
    if vocab_size <= 0 or d_model <= 0:
        raise ValueError("vocab_size dan d_model harus positif")
    return vocab_size * d_model
