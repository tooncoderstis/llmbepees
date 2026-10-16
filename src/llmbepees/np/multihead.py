"""Multi-head attention (Bab 14) — rujukan kebenaran NumPy murni.

Beberapa head menjalankan attention Bab 13 secara paralel di subruang ``d/h``,
lalu keluarannya disambung dan dicampur lewat matriks penggabung ``W_O``. Karena
tiap head dibuat lebih kecil (``d_k = d/h``), jumlah parameter **tidak** bertambah.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from llmbepees.np.attention import attention_weights
from llmbepees.np.neuron import linear

Matrix = NDArray[np.float64]


def split_heads(w: ArrayLike, num_heads: int) -> list[Matrix]:
    """Potong matriks parameter ``d × (h·d_k)`` menjadi ``h`` blok kolom (Bab 14.2)."""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"w harus 2 dimensi, dapat ndim={w.ndim}")
    if num_heads < 1:
        raise ValueError("num_heads harus >= 1")
    if w.shape[1] % num_heads != 0:
        raise ValueError(f"{w.shape[1]} kolom tidak habis dibagi {num_heads} head")
    d_k = w.shape[1] // num_heads
    return [w[:, i * d_k : (i + 1) * d_k] for i in range(num_heads)]


def concat_heads(heads: ArrayLike) -> Matrix:
    """Sambung keluaran head secara mendatar: ``Concat(head_1, …, head_h)`` (Bab 14.5)."""
    arrays = [np.asarray(h, dtype=np.float64) for h in heads]
    if not arrays:
        raise ValueError("heads tidak boleh kosong")
    return np.concatenate(arrays, axis=-1)


def multi_head_attention(
    H: ArrayLike,
    w_q: ArrayLike,
    w_k: ArrayLike,
    w_v: ArrayLike,
    w_o: ArrayLike,
    *,
    num_heads: int,
    mask: ArrayLike | None = None,
) -> tuple[Matrix, list[Matrix], list[Matrix]]:
    """Multi-head attention lengkap (Bab 14.6).

    ``head_i = Attention(HW_Qᵢ, HW_Kᵢ, HW_Vᵢ)``; ``MultiHead(H) = Concat(head)` W_O``.

    Mengembalikan ``(O, head_outputs, head_weights)``.
    """
    q_heads = split_heads(w_q, num_heads)
    k_heads = split_heads(w_k, num_heads)
    v_heads = split_heads(w_v, num_heads)

    outputs: list[Matrix] = []
    weights: list[Matrix] = []
    for q_w, k_w, v_w in zip(q_heads, k_heads, v_heads, strict=True):
        q = linear(H, q_w)
        k = linear(H, k_w)
        v = linear(H, v_w)
        a = attention_weights(q, k, mask=mask)
        weights.append(a)
        outputs.append(a @ v)

    return linear(concat_heads(outputs), w_o), outputs, weights


def kv_cache_bytes(
    num_layers: int,
    num_heads: int,
    seq_len: int,
    d_k: int,
    *,
    bytes_per_value: float = 2.0,
    num_kv_heads: int | None = None,
) -> float:
    """Ukuran KV cache: ``2·L·h·n·d_k·byte`` (Bab 14.9).

    Faktor 2 karena K dan V sama-sama disimpan. ``num_kv_heads`` < ``num_heads``
    memodelkan MQA (1 K/V) atau GQA (g grup K/V).
    """
    kv = num_heads if num_kv_heads is None else num_kv_heads
    return 2.0 * int(num_layers) * int(kv) * int(seq_len) * int(d_k) * float(bytes_per_value)


def kv_cache_ratio(num_heads: int, num_kv_heads: int) -> float:
    """Rasio ukuran cache relatif terhadap MHA: ``num_kv_heads / num_heads`` (Bab 14.9)."""
    if num_heads <= 0:
        raise ValueError("num_heads harus positif")
    return float(num_kv_heads) / float(num_heads)


__all__ = [
    "concat_heads",
    "kv_cache_bytes",
    "kv_cache_ratio",
    "multi_head_attention",
    "split_heads",
]
