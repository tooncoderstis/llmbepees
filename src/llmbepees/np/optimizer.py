"""Optimizer: SGD, momentum, Adam/AdamW, jadwal LR, clipping (Bab 12) — NumPy murni.

Gradien memberi tahu **arah**; optimizer memutuskan **seberapa jauh melangkah**.
Modul ini menyalin rumus buku apa adanya, plus rumus memori Adam (``16N``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from llmbepees.np.calculus import gradient_descent_step

Vector = NDArray[np.float64]


def _as_vector(x: ArrayLike, *, name: str = "x") -> Vector:
    vec = np.asarray(x, dtype=np.float64)
    if vec.ndim != 1:
        raise ValueError(f"{name} harus 1 dimensi, dapat ndim={vec.ndim}")
    return vec


def sgd_step(params: ArrayLike, grad: ArrayLike, *, lr: float) -> Vector:
    """Gradient descent: ``θ ← θ − η·g`` (Bab 12.1)."""
    return gradient_descent_step(params, grad, lr=lr)


def mini_batch_gradient(grads: ArrayLike) -> Vector:
    """Gradien mini-batch: rata-rata gradien ``B`` contoh, ``(1/B)∑∇L_i`` (Bab 12.3)."""
    g = np.asarray(grads, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"grads harus 2 dimensi (B×d), dapat ndim={g.ndim}")
    if g.shape[0] == 0:
        raise ValueError("grads tidak boleh kosong")
    return np.mean(g, axis=0)


def noise_std_after_batch(batch_std: float, batch_small: int, batch_large: int) -> float:
    """Derau gradien mini-batch berskala ``1/√B``: ``σ·√(B_kecil/B_besar)`` (Bab 12.3)."""
    return float(batch_std) * np.sqrt(float(batch_small) / float(batch_large))


def momentum_step(
    params: ArrayLike,
    grad: ArrayLike,
    velocity: ArrayLike,
    *,
    lr: float,
    beta: float = 0.9,
) -> tuple[Vector, Vector]:
    """SGD momentum: ``v ← βv + g``; ``θ ← θ − η·v`` (Bab 12.4)."""
    p = _as_vector(params, name="params")
    g = _as_vector(grad, name="grad")
    vel = _as_vector(velocity, name="velocity")
    if p.shape != g.shape or p.shape != vel.shape:
        raise ValueError(f"bentuk harus sama: {p.shape}, {g.shape}, {vel.shape}")
    new_velocity = beta * vel + g
    return p - lr * new_velocity, new_velocity


def momentum_limit(grad: float, *, beta: float = 0.9) -> float:
    """Nilai batas momentum untuk gradien konstan: ``v∞ = g/(1 − β)`` (Bab 12.4)."""
    return float(grad) / (1.0 - beta)


def adam_step(
    params: ArrayLike,
    grad: ArrayLike,
    m: ArrayLike,
    v: ArrayLike,
    *,
    t: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[Vector, Vector, Vector]:
    """Satu langkah Adam dengan koreksi bias (Bab 12.5).

    ``m`` = momen pertama (rata-rata gradien), ``v`` = momen kedua (rata-rata
    kuadrat gradien, elemen-per-elemen), ``t`` = nomor langkah (mulai 1).
    Mengembalikan ``(θ_baru, m_baru, v_baru)``.
    """
    p = _as_vector(params, name="params")
    g = _as_vector(grad, name="grad")
    m_vec = _as_vector(m, name="m")
    v_vec = _as_vector(v, name="v")
    if not (p.shape == g.shape == m_vec.shape == v_vec.shape):
        raise ValueError("bentuk params, grad, m, v harus sama")
    if t < 1:
        raise ValueError("t harus >= 1")

    new_m = beta1 * m_vec + (1.0 - beta1) * g
    new_v = beta2 * v_vec + (1.0 - beta2) * g * g
    m_hat = new_m / (1.0 - beta1**t)
    v_hat = new_v / (1.0 - beta2**t)
    new_p = p - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return new_p, new_m, new_v


def adamw_step(
    params: ArrayLike,
    grad: ArrayLike,
    m: ArrayLike,
    v: ArrayLike,
    *,
    t: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[Vector, Vector, Vector]:
    """AdamW: Adam + weight decay terpisah ``λθ`` (Bab 12.6).

    ``θ ← θ − η(m̂/(√v̂ + ε) + λθ)``.
    """
    p = _as_vector(params, name="params")
    new_p, new_m, new_v = adam_step(
        p, grad, m, v, t=t, lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon
    )
    new_p = new_p - lr * weight_decay * p
    return new_p, new_m, new_v


def gradient_clip(grad: ArrayLike, *, max_norm: float = 1.0) -> Vector:
    """Gradient clipping norma: jika ``‖g‖ > c``, ``g ← g·c/‖g‖`` (Bab 12.8)."""
    g = _as_vector(grad, name="grad")
    norm = float(np.linalg.norm(g))
    if norm > max_norm:
        return g * (max_norm / norm)
    return g


def lr_warmup(step: int, *, max_lr: float, warmup_steps: int) -> float:
    """Learning rate fase warmup (linear naik): ``η_t = η_max·t/T_w`` (Bab 12.7)."""
    if warmup_steps <= 0:
        raise ValueError("warmup_steps harus positif")
    return float(max_lr) * min(step, warmup_steps) / warmup_steps


def lr_cosine(
    step: int,
    *,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Jadwal warmup + cosine decay (Bab 12.7).

    ``η_t = η_min + ½(η_max − η_min)(1 + cos(π·(t − T_w)/(T − T_w)))`` setelah warmup.
    """
    if step < warmup_steps:
        return lr_warmup(step, max_lr=max_lr, warmup_steps=warmup_steps)
    if step >= total_steps:
        return float(min_lr)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return float(min_lr) + 0.5 * (max_lr - min_lr) * (1.0 + np.cos(np.pi * progress))


def adam_memory_bytes(num_parameters: float, *, bytes_per_value: float = 4.0) -> float:
    """Memori Adam: parameter + gradien + ``m`` + ``v`` = ``4·N·bytes`` (Bab 12.5).

    Pada fp32 (4 byte) menjadi ``16N`` byte; model 7 miliar ≈ 112 GB.
    """
    return 4.0 * float(num_parameters) * float(bytes_per_value)


__all__ = [
    "adam_memory_bytes",
    "adam_step",
    "adamw_step",
    "gradient_clip",
    "lr_cosine",
    "lr_warmup",
    "mini_batch_gradient",
    "momentum_limit",
    "momentum_step",
    "noise_std_after_batch",
    "sgd_step",
]
