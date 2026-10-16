"""Normalisasi (Bab 16) — rujukan kebenaran NumPy murni.

LayerNorm dan RMSNorm menormalkan **setiap token terpisah** sepanjang dimensi
fitur (bukan antarposisi), mengendalikan skala agar tumpukan blok bisa dilatih.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def layernorm(
    h: ArrayLike,
    gamma: ArrayLike | None = None,
    beta: ArrayLike | None = None,
    *,
    eps: float = 1e-5,
) -> Array:
    """LayerNorm per token (Bab 16.3).

    ``LN(h)_i = γ_i·(h_i − μ)/√(σ² + ε) + β_i`` dengan ``μ``/``σ²`` dihitung
    sepanjang dimensi fitur untuk tiap token sendiri-sendiri. ``γ`` dan ``β``
    opsional (default: tanpa skala/geser).
    """
    h = np.asarray(h, dtype=np.float64)
    mean = np.mean(h, axis=-1, keepdims=True)
    var = np.mean((h - mean) ** 2, axis=-1, keepdims=True)
    out = (h - mean) / np.sqrt(var + eps)
    if gamma is not None:
        out = out * np.asarray(gamma, dtype=np.float64)
    if beta is not None:
        out = out + np.asarray(beta, dtype=np.float64)
    return out


def rmsnorm(h: ArrayLike, gamma: ArrayLike | None = None, *, eps: float = 1e-5) -> Array:
    """RMSNorm per token (Bab 16.4).

    ``RMSNorm(h)_i = γ_i·h_i / √( (1/d)∑ h_j² + ε )``. Tanpa pengurangan rata-rata
    dan tanpa parameter ``β``; ``ε`` berada **di dalam** akar.
    """
    h = np.asarray(h, dtype=np.float64)
    ms = np.mean(h * h, axis=-1, keepdims=True)
    out = h / np.sqrt(ms + eps)
    if gamma is not None:
        out = out * np.asarray(gamma, dtype=np.float64)
    return out


__all__ = ["layernorm", "rmsnorm"]
