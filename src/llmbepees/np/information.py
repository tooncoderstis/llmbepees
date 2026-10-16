"""Teori informasi (Bab 6) — rujukan kebenaran NumPy murni.

Kandungan informasi, entropi, cross-entropy, KL divergence, perplexity,
loss LLM, dan bit-per-byte. Basis ``e`` = nat (default), basis ``2`` = bit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vector = NDArray[np.float64]

LN2 = float(np.log(2.0))


def as_vector(x: ArrayLike, *, name: str = "x") -> Vector:
    """Ubah ``x`` menjadi vektor float64 1 dimensi; tolak bentuk lain."""
    vec = np.asarray(x, dtype=np.float64)
    if vec.ndim != 1:
        raise ValueError(f"{name} harus 1 dimensi, dapat ndim={vec.ndim}")
    return vec


def _log(x: Vector, base: float) -> Vector:
    return np.log(x) / np.log(base)


def information(p: float, *, base: float = np.e) -> float:
    """Kandungan informasi (self-information): ``I(p) = −log p`` (Bab 6.1)."""
    if not 0.0 < p <= 1.0:
        raise ValueError("p harus di (0, 1]")
    return float(-_log(np.asarray([p], dtype=np.float64), base)[0])


def entropy(probs: ArrayLike, *, base: float = np.e) -> float:
    """Entropi: ``H(P) = −∑ P(x) log P(x)`` — rata-rata kejutan (Bab 6.2)."""
    p = as_vector(probs, name="probs")
    if np.any(p < 0.0):
        raise ValueError("probabilitas tidak boleh negatif")
    terms = np.where(p > 0.0, p * _log(np.where(p > 0.0, p, 1.0), base), 0.0)
    return float(-np.sum(terms))


def cross_entropy(p: ArrayLike, q: ArrayLike, *, base: float = np.e) -> float:
    """Cross-entropy: ``H(P, Q) = −∑ P(x) log Q(x)`` — fungsi loss LLM (Bab 6.3)."""
    p = as_vector(p, name="p")
    q = as_vector(q, name="q")
    if p.shape != q.shape:
        raise ValueError(f"bentuk harus sama: {p.shape} != {q.shape}")
    if np.any((p > 0.0) & (q <= 0.0)):
        raise ValueError("q harus positif di mana p > 0")
    terms = np.where(p > 0.0, p * _log(np.where(q > 0.0, q, 1.0), base), 0.0)
    return float(-np.sum(terms))


def kl_divergence(p: ArrayLike, q: ArrayLike, *, base: float = np.e) -> float:
    """KL divergence: ``D_KL(P‖Q) = ∑ P log(P/Q) = H(P,Q) − H(P)`` (Bab 6.4)."""
    return cross_entropy(p, q, base=base) - entropy(p, base=base)


def mean_negative_log_likelihood(probs: ArrayLike, *, base: float = np.e) -> float:
    """Loss rata-rata: ``−(1/N) ∑ log P(x_benar)`` (Bab 6.6)."""
    p = as_vector(probs, name="probs")
    if p.shape[0] == 0:
        raise ValueError("probs tidak boleh kosong")
    if np.any((p <= 0.0) | (p > 1.0)):
        raise ValueError("setiap probabilitas harus di (0, 1]")
    return float(-np.mean(_log(p, base)))


def perplexity(loss: float, *, base: float = np.e) -> float:
    """Perplexity: ``PPL = exp(L)`` dengan ``L`` dalam nat (Bab 6.7)."""
    if base == np.e:
        return float(np.exp(loss))
    return float(base**loss)


def bits_per_byte(total_nats: float, n_bytes: int) -> float:
    """Bit per byte: ``BPB = L_total(nat) / (ln 2 × jumlah byte)`` (Bab 6.8)."""
    if n_bytes <= 0:
        raise ValueError("n_bytes harus positif")
    return float(total_nats) / (LN2 * n_bytes)
