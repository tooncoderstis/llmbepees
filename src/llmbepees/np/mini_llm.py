"""Mini-LLM 188 parameter (Bab 21) — reproduksi eksperimen dari nol, NumPy murni.

Transformer terkecil yang masih utuh: ``V=8, d=4, L=1, head=1, d_ff=8``, panjang
maksimum 7, positional embedding **terlatih**, LayerNorm **Post-LN tanpa γ/β**,
aktivasi ReLU, **weight tying**. Inisialisasi ``N(0, 0,5)`` seed 42, gradien
**beda-hingga terpusat**, Adam lr 0,05, 400 langkah, float64.

Eksperimen ini adalah alat verifikasi: ia hanya meniru proporsi data
(``P(meja) ≈ 0,75``, ``P(tikar) ≈ 0,25``), bukan model yang bisa mengobrol.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Matrix = NDArray[np.float64]
Params = dict[str, Matrix]

VOCAB = 8
D_MODEL = 4
D_FF = 8
MAX_LEN = 7
SEED = 42
STD = 0.5
EPS = 1e-5
LR = 0.05
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1e-8

PARAM_SHAPES: list[tuple[str, tuple[int, ...]]] = [
    ("E", (VOCAB, D_MODEL)),
    ("P", (MAX_LEN, D_MODEL)),
    ("Wq", (D_MODEL, D_MODEL)),
    ("Wk", (D_MODEL, D_MODEL)),
    ("Wv", (D_MODEL, D_MODEL)),
    ("Wo", (D_MODEL, D_MODEL)),
    ("W1", (D_MODEL, D_FF)),
    ("W2", (D_FF, D_MODEL)),
]

SENTENCE_MEJA = [0, 1, 2, 3, 4, 5, 7]
SENTENCE_TIKAR = [0, 1, 2, 3, 4, 6, 7]
TRAIN_SEQUENCES = np.array(
    [SENTENCE_MEJA, SENTENCE_MEJA, SENTENCE_MEJA, SENTENCE_TIKAR], dtype=np.int64
)
CHECKPOINT_STEPS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400)


def init_params(seed: int = SEED) -> Params:
    """Parameter awal acak ``N(0, 0,5²)``; urutan E, P, Wq, Wk, Wv, Wo, W1, W2."""
    rng = np.random.default_rng(seed)
    return {name: rng.normal(0.0, STD, shape) for name, shape in PARAM_SHAPES}


def params_count(params: Params) -> int:
    """Jumlah total parameter (constant 188 untuk mini-LLM)."""
    return int(sum(v.size for v in params.values()))


def _layernorm(x: Matrix, eps: float = EPS) -> Matrix:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def forward(params: Params, ids: ArrayLike) -> Matrix:
    """Jalur maju mini-LLM; ``ids`` ``(n,)`` atau ``(batch, n)`` → logit.

    Mengembalikan logit ``(n, V)`` (atau ``(batch, n, V)``) berbentuk ``H(2)·Eᵀ``.
    """
    x = np.asarray(ids)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    x = x.astype(np.int64)
    batch, n = x.shape
    if n > MAX_LEN:
        raise ValueError(f"panjang urutan {n} melebihi MAX_LEN={MAX_LEN}")

    h = params["E"][x] + params["P"][None, :n, :]
    q = h @ params["Wq"]
    k = h @ params["Wk"]
    v = h @ params["Wv"]
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(D_MODEL)
    allowed = np.tril(np.ones((n, n), dtype=bool))
    scores = scores + np.where(allowed, 0.0, -1e9)
    scores = scores - scores.max(axis=-1, keepdims=True)
    attention = np.exp(scores)
    attention = attention / attention.sum(axis=-1, keepdims=True)

    after_attn = _layernorm(h + attention @ v @ params["Wo"])
    hidden = np.maximum(0.0, after_attn @ params["W1"])
    final_state = _layernorm(after_attn + hidden @ params["W2"])
    logits = final_state @ params["E"].T
    return logits[0] if single else logits


def sequence_loss(params: Params, sequences: ArrayLike) -> float:
    """Loss rata-rata per token atas semua posisi ber-target (Bab 18.1, 21.9)."""
    x = np.asarray(sequences, dtype=np.int64)
    logits = forward(params, x)
    targets = x[:, 1:]
    z = logits[:, :-1, :]
    z = z - z.max(axis=-1, keepdims=True)
    probs = np.exp(z)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    rows = np.arange(x.shape[0])[:, None]
    cols = np.arange(targets.shape[1])[None, :]
    selected = probs[rows, cols, targets]
    return float(-np.log(selected).mean())


def finite_difference_gradients(params: Params, sequences: ArrayLike, *, h: float = 1e-6) -> Params:
    """Gradien tiap parameter lewat beda-hingga terpusat (Bab 21.4)."""
    grads: Params = {}
    for name, value in params.items():
        flat = value.ravel()
        grad = np.empty_like(flat)
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + h
            plus = sequence_loss(params, sequences)
            flat[i] = old - h
            minus = sequence_loss(params, sequences)
            flat[i] = old
            grad[i] = (plus - minus) / (2.0 * h)
        grads[name] = grad.reshape(value.shape)
    return grads


def train(
    *,
    steps: int = 400,
    lr: float = LR,
    seed: int = SEED,
    h: float = 1e-6,
    sequences: ArrayLike | None = None,
) -> tuple[Params, dict[int, float]]:
    """Latih mini-LLM dari nol; kembalikan ``(params, riwayat_loss)`` (Bab 21)."""
    data = TRAIN_SEQUENCES if sequences is None else np.asarray(sequences, dtype=np.int64)
    params = init_params(seed)
    m = {name: np.zeros_like(v) for name, v in params.items()}
    v = {name: np.zeros_like(value) for name, value in params.items()}
    history = {0: sequence_loss(params, data)}

    for step in range(1, steps + 1):
        grads = finite_difference_gradients(params, data, h=h)
        for name in params:
            g = grads[name]
            m[name] = BETA1 * m[name] + (1.0 - BETA1) * g
            v[name] = BETA2 * v[name] + (1.0 - BETA2) * g * g
            m_hat = m[name] / (1.0 - BETA1**step)
            v_hat = v[name] / (1.0 - BETA2**step)
            params[name] = params[name] - lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
        if step in CHECKPOINT_STEPS:
            history[step] = sequence_loss(params, data)
    return params, history


def next_token_probs(params: Params, context: ArrayLike) -> Matrix:
    """Distribusi probabilitas token berikutnya untuk sebuah konteks (Bab 21.6)."""
    logits = forward(params, context)
    z = logits[-1]
    z = z - z.max()
    probs = np.exp(z)
    return probs / probs.sum()


def generate(params: Params, prompt: ArrayLike, *, max_new_tokens: int = 10) -> list[int]:
    """Generasi autoregresif greedy sampai ``<eos>`` (id 7) atau batas panjang (Bab 21.6)."""
    tokens = list(np.asarray(prompt, dtype=np.int64))
    for _ in range(max_new_tokens):
        if len(tokens) >= MAX_LEN:
            break
        probs = next_token_probs(params, tokens)
        nxt = int(np.argmax(probs))
        tokens.append(nxt)
        if nxt == 7:
            break
    return tokens


__all__ = [
    "CHECKPOINT_STEPS",
    "D_FF",
    "D_MODEL",
    "MAX_LEN",
    "SEED",
    "SENTENCE_MEJA",
    "SENTENCE_TIKAR",
    "TRAIN_SEQUENCES",
    "VOCAB",
    "finite_difference_gradients",
    "forward",
    "generate",
    "init_params",
    "next_token_probs",
    "params_count",
    "sequence_loss",
    "train",
]
