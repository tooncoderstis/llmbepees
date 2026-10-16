"""Mesin inference untuk demo: bungkus model char-level + penjelasan langkah (Bab 19, ADR-0003).

``Engine`` memuat (atau melatih) model GPT char-level kecil dan menyediakan
``generate`` serta ``explain`` — yang terakhir menunjukkan logit/probabilitas tiap
langkah sampling agar "cara kerjanya" terlihat.
"""

from __future__ import annotations

from pathlib import Path

import torch

from llmbepees.torch.model import GPT, CharTokenizer, GPTConfig, count_parameters

DEFAULT_CORPUS = Path(__file__).resolve().parents[3] / "data" / "corpus.txt"


class Engine:
    """Pembungkus tokenizer + model untuk layanan inference."""

    def __init__(
        self,
        tokenizer: CharTokenizer,
        model: GPT,
        *,
        max_len: int = 48,
        corpus_chars: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.max_len = max_len
        self.corpus_chars = corpus_chars
        self.model.eval()

    @property
    def parameter_count(self) -> int:
        return count_parameters(self.model)

    @classmethod
    def train_default(
        cls, *, steps: int = 200, seed: int = 0, corpus: Path | None = None
    ) -> Engine:
        """Latih model char-level kecil dari korpus (untuk demo tanpa checkpoint)."""
        path = DEFAULT_CORPUS if corpus is None else corpus
        text = path.read_text(encoding="utf-8")
        tokenizer = CharTokenizer(text)
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        torch.manual_seed(seed)
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=256
        )
        model = GPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

        block_size = 48
        batch_size = 16
        model.train()
        for _ in range(steps):
            idx = torch.randint(0, len(data) - block_size - 1, (batch_size,))
            x = torch.stack([data[i : i + block_size] for i in idx])
            y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return cls(tokenizer, model, max_len=block_size, corpus_chars=len(text))

    def _encode_prompt(self, prompt: str) -> list[int]:
        ids = self.tokenizer.encode(prompt)
        if not ids:
            ids = [0]
        return ids

    def _next_step(
        self,
        tokens: list[int],
        *,
        temperature: float,
        top_k: int | None,
        generator: torch.Generator | None,
    ) -> tuple[int, list[dict[str, float | str]]]:
        context = torch.tensor([tokens[-self.max_len :]], dtype=torch.long)
        with torch.no_grad():
            logits, _ = self.model(context)
        raw = logits[0, -1]
        candidates = raw / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            values = torch.topk(candidates, min(top_k, candidates.shape[0])).values
            candidates = candidates.clone()
            candidates[candidates < values[-1]] = float("-inf")
        probs = torch.softmax(candidates, dim=-1)

        token = int(torch.multinomial(probs, 1, generator=generator))
        top = torch.topk(probs, min(top_k or 5, probs.shape[0]))
        top_list = [
            {
                "token": self.tokenizer.itos[int(i)],
                "prob": round(float(p), 6),
                "logit": round(float(raw[int(i)]), 4),
            }
            for p, i in zip(top.values, top.indices, strict=True)
        ]
        return token, top_list

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> dict:
        """Hasilkan teks + jejak langkah sampling (Bab 19)."""
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        prompt_ids = self._encode_prompt(prompt)
        tokens = list(prompt_ids)
        steps: list[dict] = []
        for _ in range(max_new_tokens):
            token, top = self._next_step(
                tokens, temperature=temperature, top_k=top_k, generator=generator
            )
            tokens.append(token)
            steps.append(
                {
                    "chosen": self.tokenizer.itos[token],
                    "top": top,
                    "context": self.tokenizer.decode(tokens[-self.max_len : -1]),
                }
            )
        return {
            "prompt": prompt,
            "text": self.tokenizer.decode(tokens),
            "generated": self.tokenizer.decode(tokens[len(prompt_ids) :]),
            "steps": steps,
        }

    def explain(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 12,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> dict:
        """Seperti ``generate`` tetapi greedy (mudah dijelaskan) + probabilitas tiap langkah."""
        generator = torch.Generator().manual_seed(0)
        prompt_ids = self._encode_prompt(prompt)
        tokens = list(prompt_ids)
        steps: list[dict] = []
        for _ in range(max_new_tokens):
            token, top = self._next_step(
                tokens, temperature=temperature, top_k=top_k, generator=generator
            )
            tokens.append(token)
            steps.append(
                {
                    "position": len(tokens) - 1,
                    "context": self.tokenizer.decode(tokens[:-1]),
                    "chosen": self.tokenizer.itos[token],
                    "top": top,
                }
            )
        return {"prompt": prompt, "text": self.tokenizer.decode(tokens), "steps": steps}


__all__ = ["Engine"]
