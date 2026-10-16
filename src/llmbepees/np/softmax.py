"""Softmax, log-softmax, dan log-sum-exp (Bab 10) — rujukan kebenaran NumPy murni.

Softmax mengubah **logit** (angka sembarang, bisa negatif) menjadi **distribusi
probabilitas** (positif, jumlah 1). Versi di sini stabil secara numerik: setiap
baris dikurangi nilai maksimumnya dulu — sah karena softmax invarian terhadap
pergeseran (Bab 10.3).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vector = NDArray[np.float64]


def log_sum_exp(z: ArrayLike, *, axis: int = -1) -> Vector:
    """Log-sum-exp (LSE): ``ln ∑_j e^{z_j}`` (Bab 10.4), dihitung stabil.

    Dipakai bentuk ``max_j z_j + ln ∑_j e^{z_j − max_j z_j}`` supaya tidak meluap.
    """
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=axis, keepdims=True)
    lse = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    return np.squeeze(lse, axis=axis)


def softmax(z: ArrayLike, *, axis: int = -1) -> Vector:
    """Softmax: ``softmax(z)_i = e^{z_i} / ∑_j e^{z_j}`` (Bab 10.2), stabil.

    Implementasi mengurangkan ``max_j z_j`` dulu sehingga ``e^{z − max}`` tidak
    meluap; hasilnya sama persis karena sifat invarian pergeseran (Bab 10.3).
    """
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=axis, keepdims=True)
    e = np.exp(z - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(z: ArrayLike, *, axis: int = -1) -> Vector:
    """Log-softmax: ``z − LSE(z)`` (Bab 10.4), stabil tanpa pembagian."""
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=axis, keepdims=True)
    return (z - m) - np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))


__all__ = ["log_softmax", "log_sum_exp", "softmax"]
