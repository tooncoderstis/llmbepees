"""Fungsi kerugian di ujung jaringan (Bab 10) — rujukan kebenaran NumPy murni.

Menyatukan softmax dengan cross-entropy: loss ``L = LSE(z) − z_c`` dan — yang
paling penting — gradiennya yang sangat sederhana ``∂L/∂z = p − y`` (Bab 10.4–10.5).
Juga label smoothing (Bab 10.7) dan pembanding MSE (Bab 10.6).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from llmbepees.np.embedding import one_hot
from llmbepees.np.softmax import log_sum_exp, softmax

Vector = NDArray[np.float64]


def _logits(z: ArrayLike, *, name: str = "logits") -> Vector:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError(f"{name} harus 1 dimensi, dapat ndim={z.ndim}")
    if z.shape[0] == 0:
        raise ValueError(f"{name} tidak boleh kosong")
    return z


def _check_target(target: int, num_classes: int) -> None:
    if not 0 <= target < num_classes:
        raise ValueError(f"target {target} di luar rentang [0, {num_classes})")


def softmax_cross_entropy(logits: ArrayLike, target: int) -> float:
    """Loss softmax + cross-entropy: ``L = LSE(z) − z_c`` (Bab 10.4).

    Setara dengan ``−ln softmax(z)_c``, tetapi dihitung dalam bentuk LSE yang
    lebih stabil secara numerik (tidak butuh pembagian).
    """
    z = _logits(logits)
    _check_target(target, z.shape[0])
    return float(log_sum_exp(z) - z[target])


def softmax_cross_entropy_gradient(logits: ArrayLike, target: int) -> Vector:
    """Gradien loss terhadap logit: ``∂L/∂z = p − y`` (Bab 10.5).

    ``p = softmax(z)`` dan ``y`` = vektor one-hot label. Rumus ini adalah hasil
    penyederhanaan turunan softmax + cross-entropy; komponennya selalu berjumlah 0.
    """
    z = _logits(logits)
    _check_target(target, z.shape[0])
    p = softmax(z)
    y = one_hot(target, z.shape[0])
    return p - y


def label_smoothing(target: int, num_classes: int, epsilon: float) -> Vector:
    """Label smoothing: ``y_smooth = (1 − ε)·y + ε/V`` (Bab 10.7).

    Sebaran ``ε`` dibagi rata ke **seluruh** ``V`` kelas (termasuk kelas target).
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon harus di [0, 1]")
    y = one_hot(target, num_classes)
    return (1.0 - epsilon) * y + epsilon / num_classes


def softmax_mse_loss(logits: ArrayLike, target: int) -> float:
    """Loss MSE lewat softmax: ``∑_i (p_i − y_i)²`` (Bab 10.6, pembanding)."""
    z = _logits(logits)
    _check_target(target, z.shape[0])
    p = softmax(z)
    y = one_hot(target, z.shape[0])
    return float(np.sum((p - y) ** 2))


def softmax_mse_gradient(logits: ArrayLike, target: int) -> Vector:
    """Gradien loss MSE terhadap logit (Bab 10.6).

    ``∂L_MSE/∂z_j = 2 p_j [(p_j − y_j) − ∑_i (p_i − y_i) p_i]``.

    Faktor ``p_j`` di depan seluruh kurung membuat gradien **lenyap** saat model
    sangat salah (``p_c → 0``) — alasan MSE ditinggalkan untuk prediksi token.
    """
    z = _logits(logits)
    _check_target(target, z.shape[0])
    p = softmax(z)
    y = one_hot(target, z.shape[0])
    s = float(np.sum((p - y) * p))
    return 2.0 * p * ((p - y) - s)


__all__ = [
    "label_smoothing",
    "softmax_cross_entropy",
    "softmax_cross_entropy_gradient",
    "softmax_mse_gradient",
    "softmax_mse_loss",
]
