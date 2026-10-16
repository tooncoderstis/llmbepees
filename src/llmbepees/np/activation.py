"""Katalog fungsi aktivasi (Bab 9) — rujukan kebenaran NumPy murni.

Bab 9 memperkenalkan katalog aktivasi yang dipakai blok feed-forward:
ReLU, sigmoid, dan tanh (sudah ada di :mod:`llmbepees.np.calculus` sejak Bab 7),
serta GELU dan SiLU/Swish yang khas Transformer. Modul ini menyatukan semuanya
di satu tempat sehingga lapisan di :mod:`llmbepees.np.neuron` cukup memilih
berdasarkan nama.

ReLU/sigmoid/tanh **tidak ditulis ulang** — diimpor dari ``calculus`` agar tetap
satu sumber kebenaran.
"""

from __future__ import annotations

from math import erf

import numpy as np
from numpy.typing import ArrayLike, NDArray

from llmbepees.np.calculus import (
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    tanh_derivative,
)

Vector = NDArray[np.float64]

_erf = np.vectorize(erf, otypes=[np.float64])


def tanh(x: ArrayLike) -> Vector:
    """Tanh: ``(e^x − e^{−x}) / (e^x + e^{−x})``, keluaran di ``(−1, 1)`` (Bab 9.4)."""
    return np.tanh(np.asarray(x, dtype=np.float64))


def gelu(x: ArrayLike) -> Vector:
    """GELU eksak: ``GELU(x) = x·Φ(x)``, Φ = CDF normal standar (Bab 9.4)."""
    x = np.asarray(x, dtype=np.float64)
    cdf = 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))
    return x * cdf


def gelu_approx(x: ArrayLike) -> Vector:
    """Hampiran tanh GELU yang dipakai GPT-2/GPT-3/BERT (Bab 9.4)."""
    x = np.asarray(x, dtype=np.float64)
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_derivative(x: ArrayLike) -> Vector:
    """Turunan GELU eksak: ``Φ(x) + x·φ(x)``, φ = PDF normal standar.

    Tidak diberikan di buku; diturunkan dari definisi lalu diverifikasi numerik
    (beda hingga) di uji.
    """
    x = np.asarray(x, dtype=np.float64)
    cdf = 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    return cdf + x * pdf


def silu(x: ArrayLike) -> Vector:
    """SiLU/Swish: ``SiLU(x) = x·σ(x)``; dipakai sebagai komponen SwiGLU (Bab 9.4)."""
    x = np.asarray(x, dtype=np.float64)
    return x * sigmoid(x)


def silu_derivative(x: ArrayLike) -> Vector:
    """Turunan SiLU: ``σ(x)·(1 + x·(1 − σ(x)))`` (tidak di buku; diverifikasi numerik)."""
    x = np.asarray(x, dtype=np.float64)
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


ACTIVATIONS: dict[str, object] = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "gelu": gelu,
    "gelu_approx": gelu_approx,
    "silu": silu,
}


def get_activation(name: str) -> object:
    """Ambil fungsi aktivasi dari nama (mis. ``"relu"``, ``"gelu"``, ``"silu"``)."""
    try:
        return ACTIVATIONS[name]
    except KeyError as exc:
        raise ValueError(f"aktivasi tidak dikenal: {name!r}") from exc


__all__ = [
    "ACTIVATIONS",
    "gelu",
    "gelu_approx",
    "gelu_derivative",
    "get_activation",
    "relu",
    "relu_derivative",
    "sigmoid",
    "sigmoid_derivative",
    "silu",
    "silu_derivative",
    "tanh",
    "tanh_derivative",
]
