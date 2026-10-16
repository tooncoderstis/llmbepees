"""Pretraining: objektif, batch, dan scaling law (Bab 18) — NumPy murni.

Objektifnya sederhana: rata-rata per token dari ``−ln P_θ(x_t | x_<t)``, dengan
label adalah teks itu sendiri (self-supervised). Juga rumus batch, akumulasi
gradien, dan hukum pangkat (scaling law).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from llmbepees.np.softmax import log_softmax

Vector = NDArray[np.float64]


def shift_pairs(tokens: ArrayLike) -> tuple[Vector, Vector]:
    """Pasangan (input, target) dengan menggeser target satu posisi (Bab 18.1).

    ``input = tokens[:-1]``, ``target = tokens[1:]``.
    """
    t = np.asarray(tokens)
    if t.ndim != 1 or t.shape[0] < 2:
        raise ValueError("tokens harus vektor dengan panjang >= 2")
    return t[:-1], t[1:]


def sequence_cross_entropy(logits: ArrayLike, targets: ArrayLike) -> float:
    """Rata-rata ``−ln P(target)`` atas semua posisi (Bab 18.1).

    ``logits`` berbentuk ``(n, V)`` dan ``targets`` ``(n,)``; loss = rata-rata
    per token.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets)
    if z.ndim != 2 or y.shape[0] != z.shape[0]:
        raise ValueError(f"bentuk tidak cocok: {z.shape} vs {y.shape}")
    logp = log_softmax(z, axis=-1)
    return float(-np.mean(logp[np.arange(y.shape[0]), y.astype(np.int64)]))


def tokens_per_step(batch_size: int, seq_len: int) -> int:
    """Jumlah token per langkah pelatihan: ``B × n`` (Bab 18.3)."""
    return int(batch_size) * int(seq_len)


def gradient_accumulation(grads: ArrayLike) -> Vector:
    """Gabung ``k`` gradien sub-batch: rata-rata (``1/k ∑ g_j``) (Bab 18.3)."""
    g = np.asarray(grads, dtype=np.float64)
    if g.ndim < 1 or g.shape[0] == 0:
        raise ValueError("grads tidak boleh kosong")
    return np.mean(g, axis=0)


def random_guess_loss(vocab_size: int) -> float:
    """Loss tebak-asal saat keluaran seragam: ``ln V`` (Bab 18.1, 18.7)."""
    if vocab_size < 1:
        raise ValueError("vocab_size harus positif")
    return float(np.log(vocab_size))


def scaling_law(n: float, *, n_c: float, alpha: float) -> float:
    """Hukum pangkat ``L(N) = (N_c / N)^α`` (Bab 18.5)."""
    return float((n_c / float(n)) ** alpha)


def chinchilla_tokens(num_parameters: float, *, ratio: float = 20.0) -> float:
    """Jumlah token optimal Chinchilla: ``D ≈ ratio × N`` (Bab 18.6)."""
    return float(ratio) * float(num_parameters)


__all__ = [
    "chinchilla_tokens",
    "gradient_accumulation",
    "random_guess_loss",
    "scaling_law",
    "sequence_cross_entropy",
    "shift_pairs",
    "tokens_per_step",
]
