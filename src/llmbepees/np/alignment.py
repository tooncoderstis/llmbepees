"""Penyelarasan pasca-pretraining: SFT, reward model, DPO, LoRA (Bab 20) — NumPy murni.

Tulang punggung Transformer tidak berubah; yang berubah terutama **data** dan
**bentuk loss**. Modul ini menyalin rumus Bab 20 apa adanya.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from llmbepees.np.calculus import sigmoid

Vector = NDArray[np.float64]


def _sigmoid_scalar(x: float) -> float:
    return float(sigmoid(np.asarray(x, dtype=np.float64)))


def sft_loss(log_probs: ArrayLike, answer_mask: ArrayLike) -> float:
    """Loss SFT: rata-rata ``−ln P`` **hanya pada token jawaban** (Bab 20.2).

    ``log_probs`` = ``ln P_θ(x_t | x_<t)`` tiap posisi; ``answer_mask`` = 1 pada
    token jawaban, 0 pada token pertanyaan. Pembaginya jumlah token jawaban.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    mask = np.asarray(answer_mask, dtype=np.float64)
    if lp.shape != mask.shape:
        raise ValueError(f"bentuk harus sama: {lp.shape} != {mask.shape}")
    n_answer = float(np.sum(mask))
    if n_answer <= 0.0:
        raise ValueError("harus ada minimal satu token jawaban")
    return float(-np.sum(lp * mask) / n_answer)


def bradley_terry(r_winner: float, r_loser: float) -> float:
    """Probabilitas preferensi Bradley–Terry: ``σ(r_w − r_l)`` (Bab 20.3)."""
    return _sigmoid_scalar(float(r_winner) - float(r_loser))


def reward_model_loss(r_winner: float, r_loser: float) -> float:
    """Loss reward model: ``−ln σ(r_w − r_l)`` (Bab 20.3)."""
    return float(-np.log(bradley_terry(r_winner, r_loser)))


def rlhf_objective(reward: float, kl_divergence: float, *, beta: float) -> float:
    """Nilai objektif RLHF: ``reward − β·KL`` (Bab 20.4)."""
    return float(reward) - float(beta) * float(kl_divergence)


def dpo_loss(
    logp_winner: float,
    logp_ref_winner: float,
    logp_loser: float,
    logp_ref_loser: float,
    *,
    beta: float,
) -> float:
    """Loss DPO: ``−ln σ(β·Δln-rasio)`` (Bab 20.5).

    ``Δln-rasio = (ln π_θ(y_w) − ln π_ref(y_w)) − (ln π_θ(y_l) − ln π_ref(y_l))``.
    """
    delta_winner = float(logp_winner) - float(logp_ref_winner)
    delta_loser = float(logp_loser) - float(logp_ref_loser)
    argument = float(beta) * (delta_winner - delta_loser)
    return float(-np.log(_sigmoid_scalar(argument)))


def lora_delta(b: ArrayLike, a: ArrayLike, *, alpha: float, r: int) -> Vector:
    """Perubahan bobot LoRA: ``ΔW = (α/r)·B·A`` (Bab 20.6)."""
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    if b.ndim != 2 or a.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError(f"dimensi tidak cocok: {b.shape} · {a.shape}")
    if r <= 0:
        raise ValueError("r harus positif")
    return (float(alpha) / int(r)) * (b @ a)


def lora_parameters(d: int, k: int, r: int) -> int:
    """Parameter adapter LoRA: ``r·(d + k)`` (Bab 20.6)."""
    return int(r) * (int(d) + int(k))


__all__ = [
    "bradley_terry",
    "dpo_loss",
    "lora_delta",
    "lora_parameters",
    "reward_model_loss",
    "rlhf_objective",
    "sft_loss",
]
