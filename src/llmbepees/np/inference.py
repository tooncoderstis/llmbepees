"""Inference: sampling, temperature, top-k/p, beam, KV cache (Bab 19) — NumPy murni.

Model tetap menghasilkan distribusi probabilitas; bab ini mengatur **langkah
memilih satu token** — seberapa deterministik dan seberapa beragam keluarannya.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from llmbepees.np.multihead import kv_cache_bytes
from llmbepees.np.softmax import softmax

Vector = NDArray[np.float64]


def greedy(logits: ArrayLike) -> int:
    """Greedy decoding: ``argmax p`` — deterministik (Bab 19.2)."""
    return int(np.argmax(np.asarray(logits, dtype=np.float64)))


def temperature_softmax(logits: ArrayLike, temperature: float) -> Vector:
    """Softmax dengan temperature: ``p_i ∝ e^{z_i/T}`` (Bab 19.4)."""
    if temperature <= 0.0:
        raise ValueError("temperature harus > 0")
    return softmax(np.asarray(logits, dtype=np.float64) / float(temperature))


def top_k(probs: ArrayLike, k: int) -> Vector:
    """Top-k sampling: sisakan ``k`` teratas lalu normalisasi ulang (Bab 19.5)."""
    p = np.asarray(probs, dtype=np.float64)
    if not 1 <= k <= p.shape[0]:
        raise ValueError(f"k harus di [1, {p.shape[0]}]")
    keep = np.argsort(-p, kind="stable")[:k]
    out = np.zeros_like(p)
    out[keep] = p[keep]
    return out / np.sum(out)


def top_p(probs: ArrayLike, p: float) -> Vector:
    """Top-p (nucleus): ambil sampai kumulatif ≥ ``p`` lalu normalisasi (Bab 19.6)."""
    probs = np.asarray(probs, dtype=np.float64)
    if not 0.0 < p <= 1.0:
        raise ValueError("p harus di (0, 1]")
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    cutoff = int(np.searchsorted(cumulative, p, side="left")) + 1
    keep = order[:cutoff]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / np.sum(out)


def sample(probs: ArrayLike, rng: np.random.Generator) -> int:
    """Ambil satu indeks acak sesuai distribusi (inverse-CDF, Bab 19.3)."""
    p = np.asarray(probs, dtype=np.float64)
    return int(rng.choice(p.shape[0], p=p / np.sum(p)))


def decode_next(
    logits: ArrayLike,
    *,
    temperature: float = 1.0,
    k: int | None = None,
    p: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[int, Vector]:
    """Pipeline standar: bagi T → softmax → top-k → top-p → sampling (Bab 19.7).

    Mengembalikan ``(token, distribusi_final)``.
    """
    probs = temperature_softmax(logits, temperature)
    if k is not None:
        probs = top_k(probs, k)
    if p is not None:
        probs = top_p(probs, p)
    rng = np.random.default_rng() if rng is None else rng
    return sample(probs, rng), probs


def beam_step(
    beams: list[tuple[tuple[int, ...], float]], log_probs: ArrayLike, width: int
) -> list[tuple[tuple[int, ...], float]]:
    """Satu langkah beam search: perluas tiap beam, simpan ``width`` terbaik (Bab 19.10).

    ``beams`` = daftar ``(urutan, log-prob kumulatif)``; ``log_probs`` = matriks
    ``len(beams) × V`` berisi ``ln p`` lanjutan tiap beam.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    candidates: list[tuple[tuple[int, ...], float]] = []
    for (seq, score), row in zip(beams, lp, strict=True):
        for token, lp_token in enumerate(row):
            candidates.append((seq + (token,), score + float(lp_token)))
    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates[:width]


def prefill_work(seq_len: int, generated: int, *, use_cache: bool) -> int:
    """Total token diproses untuk menghasilkan ``generated`` token (Bab 19.8).

    Tanpa cache: ``∑(seq_len + i)``; dengan cache: ``seq_len + generated``.
    """
    if use_cache:
        return seq_len + generated
    return sum(seq_len + i for i in range(generated))


def decode_speed_upper_bound(bandwidth_gb_s: float, model_size_gb: float) -> float:
    """Batas atas token/detik saat decode: ``bandwidth / ukuran model`` (Bab 19.9)."""
    if model_size_gb <= 0.0:
        raise ValueError("model_size_gb harus positif")
    return float(bandwidth_gb_s) / float(model_size_gb)


__all__ = [
    "beam_step",
    "decode_next",
    "decode_speed_upper_bound",
    "greedy",
    "kv_cache_bytes",
    "prefill_work",
    "sample",
    "temperature_softmax",
    "top_k",
    "top_p",
]
