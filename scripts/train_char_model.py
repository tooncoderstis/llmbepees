"""Latih model GPT char-level kecil di CPU & hasilkan contoh teks (Fase 11).

Jalankan::

    .venv\\Scripts\\python scripts\\train_char_model.py

Memakai korpus ``data/corpus.txt`` (harus ada). Model sengaja kecil agar selesai
dalam puluhan detik di CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    from llmbepees.torch.model import GPT, CharTokenizer, GPTConfig, count_parameters

    text = (ROOT / "data" / "corpus.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    torch.manual_seed(0)
    config = GPTConfig(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    model = GPT(config)
    print(f"korpus     : {len(text)} karakter, vocab {tokenizer.vocab_size}")
    print(f"parameter  : {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    block_size = 48
    batch_size = 16
    steps = 300

    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in idx])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == 1:
            print(f"langkah {step:>4}  loss {loss.item():.4f}")

    model.eval()
    prompt = tokenizer.encode("Model bahasa")
    out = model.generate(prompt, max_new_tokens=160, temperature=0.8, seed=1)
    print("\ncontoh keluaran:")
    print(tokenizer.decode(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
