"""Model GPT char-level kecil (PyTorch) — "model kecil" untuk demo (ADR-0006).

Decoder-only Transformer Pre-LN dengan weight tying, mengikuti rumus Bab 16–17.
Ukurannya sengaja kecil agar bisa dilatih di CPU dan dipakai untuk demo web.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from llmbepees.torch import functional as F


@dataclass(frozen=True)
class GPTConfig:
    """Konfigurasi model char-level."""

    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    max_len: int = 64


class CharTokenizer:
    """Tokenizer karakter sederhana: setiap karakter unik menjadi satu id (Bab 8)."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention dengan proyeksi keluar ``W_O`` (Bab 13–14)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, n, d = x.shape
        d_k = d // self.n_heads
        q = self.w_q(x).view(batch, n, self.n_heads, d_k).transpose(1, 2)
        k = self.w_k(x).view(batch, n, self.n_heads, d_k).transpose(1, 2)
        v = self.w_v(x).view(batch, n, self.n_heads, d_k).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, mask=mask)
        attn = attn.transpose(1, 2).contiguous().view(batch, n, d)
        return self.w_o(attn)


class FeedForward(nn.Module):
    """FFN ``GELU(xW1)W2`` (Bab 9.6)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class Block(nn.Module):
    """Blok Transformer Pre-LN (Bab 16.2)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT decoder-only char-level (Bab 17: lima baris jalur maju)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_final = nn.LayerNorm(config.d_model)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, n = tokens.shape
        positions = torch.arange(n, device=tokens.device)
        h = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = F.causal_mask(n, device=tokens.device, dtype=h.dtype)
        for block in self.blocks:
            h = block(h, mask)
        h = self.ln_final(h)
        logits = h @ self.token_embedding.weight.T  # weight tying (Bab 17.2)

        loss = None
        if targets is not None:
            loss = F.sequence_cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt: list[int],
        *,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> list[int]:
        """Generasi autoregresif dari sebuah prompt (Bab 19)."""
        generator = None
        if seed is not None:
            generator = torch.Generator(device=next(self.parameters()).device).manual_seed(seed)

        tokens = list(prompt)
        for _ in range(max_new_tokens):
            context = torch.tensor([tokens[-self.config.max_len :]], dtype=torch.long)
            logits, _ = self(context)
            next_logits = logits[0, -1] / temperature
            if top_k is not None:
                values = torch.topk(next_logits, top_k).values
                next_logits[next_logits < values[-1]] = float("-inf")
            probs = torch.softmax(next_logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=generator))
            tokens.append(nxt)
        return tokens


def count_parameters(model: nn.Module) -> int:
    """Jumlah parameter yang dilatih (Bab 16.7)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


__all__ = [
    "Block",
    "CharTokenizer",
    "FeedForward",
    "GPT",
    "GPTConfig",
    "MultiHeadSelfAttention",
    "count_parameters",
]
