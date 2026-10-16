"""Uji model GPT char-level (Fase 11) — jalan, latih sebentar, hasilkan teks.

Dilewati bila torch tidak terpasang.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from llmbepees.torch.model import GPT, CharTokenizer, GPTConfig, count_parameters  # noqa: E402

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus.txt"


def test_tokenizer_round_trip() -> None:
    tokenizer = CharTokenizer("abcabc")
    assert tokenizer.vocab_size == 3
    assert tokenizer.decode(tokenizer.encode("cab")) == "cab"


def test_corpus_loads() -> None:
    text = CORPUS.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    assert tokenizer.vocab_size > 20
    assert tokenizer.decode(tokenizer.encode(text[:50])) == text[:50]


def test_forward_shapes_and_loss() -> None:
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=16, d_model=32, n_heads=2, n_layers=1, d_ff=64, max_len=16)
    model = GPT(config)
    x = torch.randint(0, 16, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 16)
    assert loss is not None and torch.isfinite(loss)


def test_parameter_count_is_positive() -> None:
    model = GPT(GPTConfig(vocab_size=16, d_model=32, n_heads=2, n_layers=1, d_ff=64))
    total = count_parameters(model)
    assert total > 0
    assert total == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_training_reduces_loss() -> None:
    torch.manual_seed(0)
    text = "kucing duduk di atas meja. kucing duduk di atas tikar. " * 8
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    config = GPTConfig(vocab_size=tokenizer.vocab_size, d_model=32, n_heads=2, n_layers=1, d_ff=64)
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    block = 16
    x = data[:block].unsqueeze(0)
    y = data[1 : block + 1].unsqueeze(0)

    with torch.no_grad():
        _, initial_tensor = model(x, y)
    initial = float(initial_tensor)
    for _ in range(60):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        _, final_tensor = model(x, y)
    assert float(final_tensor) < initial


def test_generate_returns_prompt_plus_tokens() -> None:
    torch.manual_seed(0)
    tokenizer = CharTokenizer("abcabc")
    model = GPT(GPTConfig(vocab_size=3, d_model=16, n_heads=1, n_layers=1, d_ff=32, max_len=8))
    prompt = tokenizer.encode("ab")
    out = model.generate(prompt, max_new_tokens=5, seed=0)
    assert out[:2] == prompt
    assert len(out) == 7
    assert isinstance(tokenizer.decode(out), str)
