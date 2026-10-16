"""Operasi inti Transformer di PyTorch (port dari ``llmbepees.np``) — ADR-0001.

Nama dan rumus sengaja dibuat sama dengan versi NumPy agar bisa diuji **paritas**
angka-per-angka. Gunakan ``dtype=torch.float64`` untuk perbandingan ketat.
"""

from __future__ import annotations

import math

import torch

Tensor = torch.Tensor


def log_sum_exp(z: Tensor, *, axis: int = -1) -> Tensor:
    """Log-sum-exp stabil (Bab 10.4)."""
    return torch.logsumexp(z, dim=axis)


def softmax(z: Tensor, *, axis: int = -1) -> Tensor:
    """Softmax per baris (Bab 10.2)."""
    return torch.softmax(z, dim=axis)


def log_softmax(z: Tensor, *, axis: int = -1) -> Tensor:
    """Log-softmax (Bab 10.4)."""
    return torch.log_softmax(z, dim=axis)


def layernorm(
    h: Tensor,
    gamma: Tensor | None = None,
    beta: Tensor | None = None,
    *,
    eps: float = 1e-5,
) -> Tensor:
    """LayerNorm per token sepanjang dimensi fitur (Bab 16.3)."""
    mean = h.mean(dim=-1, keepdim=True)
    var = ((h - mean) ** 2).mean(dim=-1, keepdim=True)
    out = (h - mean) / torch.sqrt(var + eps)
    if gamma is not None:
        out = out * gamma
    if beta is not None:
        out = out + beta
    return out


def rmsnorm(h: Tensor, gamma: Tensor | None = None, *, eps: float = 1e-5) -> Tensor:
    """RMSNorm: ``γ·h/√(mean(h²)+ε)`` (Bab 16.4)."""
    ms = (h * h).mean(dim=-1, keepdim=True)
    out = h / torch.sqrt(ms + eps)
    if gamma is not None:
        out = out * gamma
    return out


def gelu(x: Tensor) -> Tensor:
    """GELU eksak ``x·Φ(x)`` (Bab 9.4)."""
    return torch.nn.functional.gelu(x)


def gelu_approx(x: Tensor) -> Tensor:
    """Hampiran tanh GELU (Bab 9.4)."""
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + torch.tanh(inner))


def silu(x: Tensor) -> Tensor:
    """SiLU/Swish ``x·σ(x)`` (Bab 9.4)."""
    return torch.nn.functional.silu(x)


def causal_mask(n: int, *, neg_inf: float = -1e9, device=None, dtype=None) -> Tensor:
    """Masker kausal aditif ``n×n`` (Bab 13.4)."""
    allowed = torch.tril(torch.ones((n, n), device=device, dtype=torch.bool))
    return torch.where(
        allowed,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), float(neg_inf), device=device, dtype=dtype),
    )


def attention_weights(q: Tensor, k: Tensor, *, mask: Tensor | None = None) -> Tensor:
    """Bobot attention ``softmax(QKᵀ/√d_k + M)`` (Bab 13.4, 13.6)."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores + mask
    return torch.softmax(scores, dim=-1)


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, *, mask: Tensor | None = None
) -> Tensor:
    """Scaled dot-product attention ``softmax(QKᵀ/√d_k + M)·V`` (Bab 13.6)."""
    return attention_weights(q, k, mask=mask) @ v


def split_heads(w: Tensor, num_heads: int) -> list[Tensor]:
    """Potong matriks parameter menjadi ``num_heads`` blok kolom (Bab 14.2)."""
    if w.shape[1] % num_heads != 0:
        raise ValueError(f"{w.shape[1]} kolom tidak habis dibagi {num_heads} head")
    d_k = w.shape[1] // num_heads
    return [w[:, i * d_k : (i + 1) * d_k] for i in range(num_heads)]


def multi_head_attention(
    h: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    *,
    num_heads: int,
    mask: Tensor | None = None,
) -> tuple[Tensor, list[Tensor]]:
    """Multi-head attention ``Concat(head_i)·W_O`` (Bab 14.6)."""
    q_heads = split_heads(w_q, num_heads)
    k_heads = split_heads(w_k, num_heads)
    v_heads = split_heads(w_v, num_heads)
    outputs = []
    for q_w, k_w, v_w in zip(q_heads, k_heads, v_heads, strict=True):
        q = h @ q_w
        k = h @ k_w
        v = h @ v_w
        outputs.append(scaled_dot_product_attention(q, k, v, mask=mask))
    concatenated = torch.cat(outputs, dim=-1)
    return concatenated @ w_o, outputs


def softmax_cross_entropy(logits: Tensor, target: int) -> Tensor:
    """Loss ``LSE(z) − z_c`` untuk satu posisi (Bab 10.4)."""
    return torch.logsumexp(logits, dim=-1) - logits[target]


def sequence_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Rata-rata ``−ln P(target)`` atas semua posisi (Bab 18.1)."""
    return torch.nn.functional.cross_entropy(logits, targets.long())


__all__ = [
    "attention_weights",
    "causal_mask",
    "gelu",
    "gelu_approx",
    "layernorm",
    "log_softmax",
    "log_sum_exp",
    "multi_head_attention",
    "rmsnorm",
    "scaled_dot_product_attention",
    "sequence_cross_entropy",
    "silu",
    "softmax",
    "softmax_cross_entropy",
    "split_heads",
]
