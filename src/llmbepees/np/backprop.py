"""Backpropagation (Bab 11) — rujukan kebenaran NumPy murni.

Aturan mundur tiap jenis operasi, contoh jaringan mungil Bab 11.3 (maju lalu
mundur, lengkap dengan pembaruan parameter), residual connection, backprop
untuk banyak token, serta rumus biaya FLOP (``≈ 6ND``) dan memori aktivasi.

Prinsipnya cuma aturan rantai Bab 7 yang dijalankan mundur — dari loss ke
masukan — dan disusun ulang supaya hasil antara tetap berukuran vektor.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from llmbepees.np.calculus import relu_derivative

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


def add_backward(grad_out: ArrayLike) -> Vector:
    """Mundur untuk ``y = x + b``: gradien lewat begitu saja (Bab 11.4)."""
    return np.asarray(grad_out, dtype=np.float64)


def relu_backward(x: ArrayLike, grad_out: ArrayLike) -> Vector:
    """Mundur ReLU: ``gout ⊙ 1[x > 0]`` (Bab 11.4)."""
    return np.asarray(grad_out, dtype=np.float64) * relu_derivative(x)


def hadamard_backward(a: ArrayLike, b: ArrayLike, grad_out: ArrayLike) -> tuple[Vector, Vector]:
    """Mundur ``y = a ⊙ b``: ``(gout ⊙ b, gout ⊙ a)`` (Bab 11.4)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    g = np.asarray(grad_out, dtype=np.float64)
    return g * b, g * a


def matmul_backward(x: ArrayLike, w: ArrayLike, grad_out: ArrayLike) -> tuple[Matrix, Matrix]:
    """Mundur ``y = xW``: ``grad_W = Xᵀ G`` dan ``grad_X = G Wᵀ`` (Bab 11.4, 11.6).

    ``x`` boleh vektor ``(d,)`` atau batch ``(n, d)``; ``w`` ``(d, m)``;
    ``grad_out`` mengikuti bentuk keluaran ``y``. Untuk batch, kontribusi semua
    ``n`` token dijumlahkan ke satu ``grad_W`` berukuran ``(d, m)``.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    g = np.asarray(grad_out, dtype=np.float64)
    if x.ndim == 1:
        grad_w = np.outer(x, g)
        grad_x = g @ w.T
    elif x.ndim == 2:
        grad_w = x.T @ g
        grad_x = g @ w.T
    else:
        raise ValueError(f"x harus 1 atau 2 dimensi, dapat ndim={x.ndim}")
    return grad_w, grad_x


def residual_backward(grad_out: ArrayLike) -> Vector:
    """Jalur identitas residual ``y = x + F(x)``: gradien lewat tanpa dikali (Bab 11.5)."""
    return np.asarray(grad_out, dtype=np.float64)


def residual_gradient(grad_out: ArrayLike, jacobian_f: ArrayLike) -> Vector:
    """Gradien penuh residual: ``grad_out + J_Fᵀ·grad_out`` (Bab 11.5).

    Suku pertama (identitas ``I``) adalah jalur yang tidak pernah dikalikan faktor
    apa pun — itulah yang mencegah gradien meluruh habis karena kedalaman.
    """
    g = np.asarray(grad_out, dtype=np.float64)
    j = np.asarray(jacobian_f, dtype=np.float64)
    return g + j.T @ g


def forward_mlp(x: float, w1: float, b1: float, w2: float, b2: float) -> tuple[float, float, float]:
    """Maju jaringan mungil Bab 11.3: ``z1 → h=ReLU(z1) → y``."""
    z1 = np.float64(w1) * np.float64(x) + np.float64(b1)
    h = np.maximum(0.0, z1)
    y = np.float64(w2) * h + np.float64(b2)
    return float(z1), float(h), float(y)


def two_layer_mlp_backward(
    x: float, w1: float, b1: float, w2: float, b2: float, target: float
) -> dict[str, float]:
    """Maju + mundur lengkap jaringan mungil Bab 11.3 (loss ``L = (y − t)²``).

    Mengembalikan semua nilai antara hasil maju (``z1``, ``h``, ``y``, ``loss``)
    dan seluruh gradien (``dL_dy``, ``dL_dw2``, ``dL_db2``, ``dL_dh``, ``dL_dz1``,
    ``dL_dw1``, ``dL_db1``). Satu langkah mundur memakai hasil langkah sebelumnya.
    """
    z1, h, y = forward_mlp(x, w1, b1, w2, b2)
    loss = (y - target) ** 2

    dL_dy = 2.0 * (y - target)
    dL_dw2 = dL_dy * h
    dL_db2 = dL_dy
    dL_dh = dL_dy * w2
    dL_dz1 = dL_dh * float(np.asarray(relu_derivative(z1), dtype=np.float64))
    dL_dw1 = dL_dz1 * x
    dL_db1 = dL_dz1

    return {
        "z1": z1,
        "h": h,
        "y": y,
        "loss": loss,
        "dL_dy": dL_dy,
        "dL_dw2": dL_dw2,
        "dL_db2": dL_db2,
        "dL_dh": dL_dh,
        "dL_dz1": dL_dz1,
        "dL_dw1": dL_dw1,
        "dL_db1": dL_db1,
    }


def training_flops(num_parameters: float, num_tokens: float) -> float:
    """Biaya pelatihan ``C ≈ 6·N·D`` FLOP (maju + mundur) (Bab 11.7)."""
    return 6.0 * float(num_parameters) * float(num_tokens)


def training_time_seconds(flops: float, flops_per_second: float) -> float:
    """Perkiraan waktu latih: ``FLOP / throughput`` detik (Bab 11.7)."""
    if flops_per_second <= 0.0:
        raise ValueError("flops_per_second harus positif")
    return float(flops) / float(flops_per_second)


def activation_memory(
    num_layers: int,
    seq_len: int,
    d_model: int,
    *,
    k: float = 12.0,
    bytes_per_value: float = 2.0,
) -> float:
    """Memori aktivasi ``≈ k·L·n·d·bytes`` (Bab 11.8)."""
    return float(k) * int(num_layers) * int(seq_len) * int(d_model) * float(bytes_per_value)


def factor_power(factor: float, depth: int) -> float:
    """Hasil kali ``depth`` faktor — gradien tanpa residual (Bab 11.5).

    ``0,8^96 ≈ 5·10⁻¹⁰`` (gradien lenyap) dan ``1,2^96 ≈ 4·10⁷`` (gradien meledak).
    """
    return float(factor) ** int(depth)


__all__ = [
    "activation_memory",
    "add_backward",
    "factor_power",
    "forward_mlp",
    "hadamard_backward",
    "matmul_backward",
    "relu_backward",
    "residual_backward",
    "residual_gradient",
    "training_flops",
    "training_time_seconds",
    "two_layer_mlp_backward",
]
