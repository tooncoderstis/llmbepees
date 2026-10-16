"""Blok Transformer (Bab 16) — rujukan kebenaran NumPy murni.

Satu blok = dua sub-blok, masing-masing dibungkus residual: attention (tukar
informasi antarposisi) lalu FFN (olah informasi di dalam posisi). Ukuran matriks
masuk = keluar (``n×d``), sehingga blok yang sama bisa ditumpuk berulang.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from llmbepees.np.layernorm import layernorm

Matrix = NDArray[np.float64]
Block = Callable[[Matrix], Matrix]


def pre_ln_block(H: ArrayLike, mha: Block, ffn: Block) -> Matrix:
    """Blok Pre-LN (Bab 16.2): normalisasi **sebelum** tiap sub-blok.

    ``H'  = H + MHA(LN(H))`` dan ``H'' = H' + FFN(LN(H'))``.
    """
    h = np.asarray(H, dtype=np.float64)
    a = h + mha(layernorm(h))
    return a + ffn(layernorm(a))


def post_ln_block(H: ArrayLike, mha: Block, ffn: Block) -> Matrix:
    """Blok Post-LN (Bab 16.5): normalisasi **setelah** penjumlahan residual.

    ``H' = LN(H + MHA(H))`` dan ``H'' = LN(H' + FFN(H'))``. Bentuk ini dipakai
    mini-LLM karena dengan satu lapisan ia lebih mudah ditelusuri.
    """
    h = np.asarray(H, dtype=np.float64)
    a = layernorm(h + mha(h))
    return layernorm(a + ffn(a))


def stack_blocks(H: ArrayLike, blocks: Iterable[Block]) -> Matrix:
    """Tumpuk blok berurutan: ``H(ℓ) = Block_ℓ(H(ℓ−1))`` (Bab 16.6)."""
    h = np.asarray(H, dtype=np.float64)
    for block in blocks:
        h = block(h)
    return h


def attention_block_parameters(d: int) -> int:
    """Parameter attention per blok: ``W_Q, W_K, W_V, W_O`` masing-masing ``d×d`` (Bab 16.7)."""
    return 4 * d * d


def ffn_block_parameters(d: int, d_ff: int) -> int:
    """Parameter FFN per blok: ``W1`` ``d×d_ff`` dan ``W2`` ``d_ff×d`` (Bab 16.7)."""
    return 2 * d * d_ff


def block_parameters(d: int, d_ff: int) -> int:
    """Total parameter satu blok: ``4d² + 2·d·d_ff + 4d`` (2×LayerNorm) (Bab 16.7).

    Dengan ``d_ff = 4d`` menjadi ``12d² + 4d``; mengabaikan suku kecil: ``≈12d²``.
    """
    return attention_block_parameters(d) + ffn_block_parameters(d, d_ff) + 4 * d


__all__ = [
    "attention_block_parameters",
    "block_parameters",
    "ffn_block_parameters",
    "post_ln_block",
    "pre_ln_block",
    "stack_blocks",
]
