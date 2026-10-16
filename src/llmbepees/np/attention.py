"""Attention: query, key, value, scaled dot-product (Bab 13) — NumPy murni.

Attention memindahkan informasi **antarposisi**: setiap token membuat query (yang
dicari), key (yang ditawarkan), dan value (isi yang diambil); kecocokan query–key
menjadi bobot lewat softmax, lalu bobot itu mencampur value.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from llmbepees.np.neuron import linear
from llmbepees.np.softmax import softmax

Matrix = NDArray[np.float64]


def _as_matrix(x: ArrayLike, *, name: str = "x") -> Matrix:
    mat = np.asarray(x, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"{name} harus 2 dimensi, dapat ndim={mat.ndim}")
    return mat


def causal_mask(n: int, *, neg_inf: float = -1e9) -> Matrix:
    """Masker kausal aditif ``n×n``: 0 pada ``j ≤ i``, ``neg_inf`` pada ``j > i`` (Bab 13.4).

    Dipakai sebelum softmax agar posisi hanya melihat ke belakang. Buku memakai
    ``−∞``; implementasi nyata memakai angka sangat negatif (mis. ``−1e9``) supaya
    ``e^{−∞} ≈ 0`` tanpa menghasilkan ``NaN``.
    """
    if n <= 0:
        raise ValueError("n harus positif")
    allowed = np.tril(np.ones((n, n), dtype=bool))
    return np.where(allowed, 0.0, float(neg_inf))


def project_qkv(H: ArrayLike, w_q: ArrayLike, w_k: ArrayLike, w_v: ArrayLike) -> tuple:
    """Bentuk query/key/value: ``Q = HW_Q``, ``K = HW_K``, ``V = HW_V`` (Bab 13.2)."""
    return linear(H, w_q), linear(H, w_k), linear(H, w_v)


def attention_scores(q: ArrayLike, k: ArrayLike) -> Matrix:
    """Skor kecocokan semua pasangan: ``S = QKᵀ`` (Bab 13.3). Bentuk ``n×n``."""
    q = _as_matrix(q, name="q")
    k = _as_matrix(k, name="k")
    if q.shape[1] != k.shape[1]:
        raise ValueError(f"dimensi tidak cocok: {q.shape} · {k.shape}")
    return q @ k.T


def attention_weights(q: ArrayLike, k: ArrayLike, *, mask: ArrayLike | None = None) -> Matrix:
    """Bobot attention per baris: ``softmax(QKᵀ/√d_k + M)`` (Bab 13.4, 13.6).

    Pembagian ``√d_k`` menjaga varians skor tetap ≈ 1; ``mask`` (opsional) adalah
    masker kausal aditif dari :func:`causal_mask`.
    """
    q = _as_matrix(q, name="q")
    scores = attention_scores(q, k) / np.sqrt(q.shape[1])
    if mask is not None:
        scores = scores + np.asarray(mask, dtype=np.float64)
    return softmax(scores, axis=-1)


def scaled_dot_product_attention(
    q: ArrayLike, k: ArrayLike, v: ArrayLike, *, mask: ArrayLike | None = None
) -> Matrix:
    """Scaled dot-product attention: ``softmax(QKᵀ/√d_k + M)·V`` (Bab 13.6).

    Keluaran untuk sebuah posisi adalah **rata-rata berbobot** dari seluruh vektor
    value; bobot tiap baris berjumlah 1.
    """
    weights = attention_weights(q, k, mask=mask)
    return weights @ _as_matrix(v, name="v")


def single_head_attention(
    H: ArrayLike, w_q: ArrayLike, w_k: ArrayLike, w_v: ArrayLike, *, mask: ArrayLike | None = None
) -> tuple[Matrix, Matrix]:
    """Attention satu head dari representasi ``H``; mengembalikan ``(O, A)`` (Bab 13)."""
    q, k, v = project_qkv(H, w_q, w_k, w_v)
    weights = attention_weights(q, k, mask=mask)
    return weights @ v, weights


def score_variance(dk: int, *, trials: int = 20_000, seed: int = 0) -> float:
    """Varians terukur dot product q·k untuk komponen normal standar (Bab 13.4).

    Secara teori tumbuh sebanding ``d_k`` — alasan pembagian ``√d_k``.
    """
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((trials, dk))
    k = rng.standard_normal((trials, dk))
    return float(np.var(np.sum(q * k, axis=1)))


def attention_flops(
    n: int, d: int, d_k: int, d_v: int, *, d_out: int | None = None
) -> dict[str, float]:
    """Perkiraan biaya FLOP attention (Bab 13.9).

    Dua suku tumbuh dengan ``n²`` (skor & pencampuran value), sisanya dengan ``n``
    (proyeksi). Mengembalikan rincian per operasi plus ``total``.
    """
    d_out = d if d_out is None else d_out
    projection = 3 * 2 * n * d * d_k
    scores = 2 * n * n * d_k
    soft = 3 * n * n
    weighted_values = 2 * n * n * d_v
    output_projection = 2 * n * d * d_out
    return {
        "projection": float(projection),
        "scores": float(scores),
        "softmax": float(soft),
        "weighted_values": float(weighted_values),
        "output_projection": float(output_projection),
        "n_squared": float(scores + soft + weighted_values),
        "n_linear": float(projection + output_projection),
        "total": float(projection + scores + soft + weighted_values + output_projection),
    }


__all__ = [
    "attention_flops",
    "attention_scores",
    "attention_weights",
    "causal_mask",
    "project_qkv",
    "scaled_dot_product_attention",
    "score_variance",
    "single_head_attention",
]
