"""Model Transformer utuh & lapisan keluaran (Bab 17) — NumPy murni.

Menyusun jalur maju lengkap: ``X + P`` → tumpukan blok → ``LN`` → logit
(``H·Eᵀ``, weight tying) → softmax. Juga rumus jumlah parameter (Bab 16.7),
ukuran matriks tiap tahap (17.6), dan hitungan operasi (17.7).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from llmbepees.np.block import block_parameters, stack_blocks
from llmbepees.np.layernorm import layernorm

Matrix = NDArray[np.float64]
Block = Callable[[Matrix], Matrix]


def output_logits(H: ArrayLike, E: ArrayLike) -> Matrix:
    """Lapisan keluaran dengan weight tying: ``Z = H·Eᵀ`` (Bab 17.2).

    ``H`` = keadaan akhir ``n×d``, ``E`` = matriks embedding ``V×d``; ``Z`` =
    logit ``n×V``. Logit token ``j`` adalah dot product keadaan akhir dan
    embedding token ``j``.
    """
    h = np.asarray(H, dtype=np.float64)
    e = np.asarray(E, dtype=np.float64)
    if h.ndim != 2 or e.ndim != 2 or h.shape[1] != e.shape[1]:
        raise ValueError(f"dimensi tidak cocok: {h.shape} · {e.shape}")
    return h @ e.T


def decoder_forward(
    H0: ArrayLike,
    blocks: Iterable[Block],
    E: ArrayLike,
    *,
    final_norm: bool = False,
) -> Matrix:
    """Jalur maju decoder-only: tumpuk blok lalu hasilkan logit (Bab 17).

    ``H0`` adalah ``X + P``. Bila ``final_norm`` benar, ``LN`` diterapkan sebelum
    lapisan keluaran (``Z = LN(H(L))·Eᵀ``, Bab 17.8).
    """
    h = stack_blocks(H0, blocks)
    if final_norm:
        h = layernorm(h)
    return output_logits(h, E)


def approx_model_parameters(V: int, d: int, n_max: int, L: int) -> int:
    """Rumus parameter model (Bab 16.7, ``d_ff=4d``): ``V·d + n_max·d + L·12d²``."""
    return V * d + n_max * d + L * 12 * d * d


def model_parameters(V: int, d: int, n_max: int, L: int, d_ff: int) -> int:
    """Parameter model dengan rumus blok penuh (tanpa bias) (Bab 16.7)."""
    return V * d + n_max * d + L * block_parameters(d, d_ff)


def predictions_per_sequence(n: int) -> int:
    """Jumlah contoh pelatihan per urutan: ``n − 1`` (Bab 17.4)."""
    return max(n - 1, 0)


def forward_macs(n: int, d: int, d_ff: int, V: int, *, num_heads: int) -> int:
    """Jumlah perkalian jalur maju mini-LLM (Bab 17.7).

    Proyeksi Q/K/V, skor & campuran attention, ``W_O``, FFN naik/turun, dan lapisan
    keluaran. Untuk mini-LLM (n=5,d=4,d_ff=8,V=8,h=2) hasilnya 1.000.
    """
    d_k = d // num_heads
    projections = 3 * n * d * d
    scores = num_heads * n * n * d_k
    weighted_values = num_heads * n * n * d_k
    output_projection = n * d * d
    ffn_up = n * d * d_ff
    ffn_down = n * d_ff * d
    logits = n * d * V
    return projections + scores + weighted_values + output_projection + ffn_up + ffn_down + logits


def forward_flops(n: int, num_parameters: float) -> float:
    """Perkiraan FLOP jalur maju: ``2·N·n`` (2 FLOP per parameter per token) (Bab 17.7)."""
    return 2.0 * float(num_parameters) * int(n)


def matrix_sizes(n: int, d: int, d_ff: int, V: int) -> dict[str, str]:
    """Ukuran matriks tiap tahap jalur maju (Bab 17.6)."""
    return {
        "token_id": f"{n}",
        "embedding_X": f"{n} x {d}",
        "qkv": f"{n} x {d}",
        "scores": f"{n} x {n}",
        "mha_output": f"{n} x {d}",
        "ffn_hidden": f"{n} x {d_ff}",
        "block_output": f"{n} x {d}",
        "logits": f"{n} x {V}",
    }


__all__ = [
    "approx_model_parameters",
    "decoder_forward",
    "forward_flops",
    "forward_macs",
    "matrix_sizes",
    "model_parameters",
    "output_logits",
    "predictions_per_sequence",
]
