"""Probabilitas (Bab 5) — rujukan kebenaran NumPy murni.

Distribusi, probabilitas bersyarat, ekspektasi, varians, distribusi normal,
dan sampling dari distribusi diskrit (inverse transform sampling).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vector = NDArray[np.float64]

TWO_PI = 2.0 * np.pi


def as_vector(x: ArrayLike, *, name: str = "x") -> Vector:
    """Ubah ``x`` menjadi vektor float64 1 dimensi; tolak bentuk lain."""
    vec = np.asarray(x, dtype=np.float64)
    if vec.ndim != 1:
        raise ValueError(f"{name} harus 1 dimensi, dapat ndim={vec.ndim}")
    return vec


def validate_distribution(probs: ArrayLike, *, atol: float = 1e-6) -> Vector:
    """Pastikan ``probs`` distribusi sah: setiap nilai ∈ [0,1] dan jumlahnya 1 (Bab 5.1)."""
    p = as_vector(probs, name="probs")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("pelanggaran A1: setiap probabilitas harus di [0, 1]")
    if not np.isclose(np.sum(p), 1.0, atol=atol):
        raise ValueError(f"pelanggaran A2: jumlah probabilitas harus 1, dapat {np.sum(p)}")
    return p


def uniform(n: int) -> Vector:
    """Distribusi seragam atas ``n`` hasil, masing-masing ``1/n`` (Bab 5.1)."""
    if n <= 0:
        raise ValueError("n harus positif")
    return np.full(n, 1.0 / n, dtype=np.float64)


def conditional_probability(p_joint: float, p_given: float) -> float:
    """Probabilitas bersyarat: ``P(A|B) = P(A ∩ B) / P(B)`` (Bab 5.2)."""
    if p_given <= 0.0:
        raise ValueError("P(B) harus positif")
    return float(p_joint) / float(p_given)


def expected_value(values: ArrayLike, probs: ArrayLike) -> float:
    """Nilai harapan: ``E[x] = ∑ P(x) · x`` (Bab 5.5)."""
    values = as_vector(values, name="values")
    probs = as_vector(probs, name="probs")
    if values.shape != probs.shape:
        raise ValueError(f"bentuk harus sama: {values.shape} != {probs.shape}")
    return float(np.sum(probs * values))


def variance(values: ArrayLike, probs: ArrayLike) -> float:
    """Varians: ``Var(x) = E[(x − E[x])²] = E[x²] − (E[x])²`` (Bab 5.6)."""
    mean = expected_value(values, probs)
    values = as_vector(values, name="values")
    probs = as_vector(probs, name="probs")
    return float(np.sum(probs * (values - mean) ** 2))


def std(values: ArrayLike, probs: ArrayLike) -> float:
    """Simpangan baku: ``σ = √Var(x)`` (Bab 5.6)."""
    return float(np.sqrt(variance(values, probs)))


def cumulative_sum(probs: ArrayLike) -> Vector:
    """Probabilitas kumulatif: ``C_k = ∑_{i=1}^{k} p_i`` (Bab 5.8)."""
    return np.cumsum(as_vector(probs, name="probs"))


def sample_categorical(
    probs: ArrayLike,
    *,
    rng: np.random.Generator | None = None,
    u: float | None = None,
) -> int:
    """Pilih indeks token dengan *inverse transform sampling* (Bab 5.8).

    Jika ``u`` diberikan (angka acak ∈ [0,1)), dipakai langsung — berguna untuk uji
    deterministik. Jika tidak, ``u`` diambil dari ``rng`` (atau generator baru).
    """
    p = validate_distribution(probs)
    if u is None:
        u = float((rng or np.random.default_rng()).random())
    if not 0.0 <= u < 1.0:
        raise ValueError("u harus di [0, 1)")
    c = np.cumsum(p)
    idx = int(np.searchsorted(c, u, side="right"))
    return min(idx, p.shape[0] - 1)


def normal_pdf(x: ArrayLike, *, mu: float = 0.0, sigma: float = 1.0) -> Vector:
    """Kepadatan distribusi normal ``N(mu, sigma²)`` (Bab 5.7)."""
    if sigma <= 0.0:
        raise ValueError("sigma harus positif")
    x = as_vector(x, name="x")
    return np.exp(-((x - mu) ** 2) / (2.0 * sigma**2)) / (sigma * np.sqrt(TWO_PI))
