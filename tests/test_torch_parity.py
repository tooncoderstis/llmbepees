"""Uji paritas NumPy ↔ PyTorch (ADR-0001) — port harus setara.

Dilewati bila torch tidak terpasang. Semua perbandingan memakai float64 → toleransi ketat.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from llmbepees.np.activation import gelu as np_gelu  # noqa: E402
from llmbepees.np.activation import gelu_approx as np_gelu_approx  # noqa: E402
from llmbepees.np.activation import silu as np_silu  # noqa: E402
from llmbepees.np.attention import attention_weights as np_attention_weights  # noqa: E402
from llmbepees.np.attention import causal_mask as np_causal_mask  # noqa: E402
from llmbepees.np.attention import project_qkv as np_project_qkv  # noqa: E402
from llmbepees.np.attention import scaled_dot_product_attention as np_sdpa  # noqa: E402
from llmbepees.np.layernorm import layernorm as np_layernorm_fn  # noqa: E402
from llmbepees.np.layernorm import rmsnorm as np_rmsnorm_fn  # noqa: E402
from llmbepees.np.loss import softmax_cross_entropy as np_softmax_ce  # noqa: E402
from llmbepees.np.multihead import multi_head_attention as np_mha  # noqa: E402
from llmbepees.np.softmax import log_softmax as np_log_softmax  # noqa: E402
from llmbepees.np.softmax import log_sum_exp as np_log_sum_exp  # noqa: E402
from llmbepees.np.softmax import softmax as np_softmax_fn  # noqa: E402
from llmbepees.np.training import sequence_cross_entropy as np_sequence_ce  # noqa: E402
from llmbepees.torch import functional as tF  # noqa: E402

TOL = 1e-6

H = [
    [0.5, 0.0, 0.0, 0.0],
    [1.0, 0.5, 1.0, 0.0],
    [0.0, 1.0, 0.5, 1.0],
    [1.0, 1.0, 0.0, 0.5],
    [0.5, 0.5, 1.0, 1.0],
]
W_Q = [[1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1]]
W_K = [[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]]
W_V = 0.5 * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
W_O = 0.5 * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]])


def _t(x) -> "torch.Tensor":
    return torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)


def test_softmax_parity() -> None:
    z = [2.0, 1.0, 0.1, -3.0]
    np.testing.assert_allclose(np_softmax_fn(z), tF.softmax(_t(z)).numpy(), atol=TOL)
    np.testing.assert_allclose(np_log_softmax(z), tF.log_softmax(_t(z)).numpy(), atol=TOL)
    assert np_log_sum_exp(z) == pytest.approx(float(tF.log_sum_exp(_t(z))), abs=TOL)


def test_layernorm_rmsnorm_parity() -> None:
    h = [[2.0, 4.0, 4.0, 6.0], [1.0, -1.0, 0.5, 0.0]]
    np.testing.assert_allclose(np_layernorm_fn(h), tF.layernorm(_t(h)).numpy(), atol=TOL)
    np.testing.assert_allclose(np_rmsnorm_fn(h), tF.rmsnorm(_t(h)).numpy(), atol=TOL)


def test_activation_parity() -> None:
    x = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    np.testing.assert_allclose(np_gelu(x), tF.gelu(_t(x)).numpy(), atol=TOL)
    np.testing.assert_allclose(np_gelu_approx(x), tF.gelu_approx(_t(x)).numpy(), atol=TOL)
    np.testing.assert_allclose(np_silu(x), tF.silu(_t(x)).numpy(), atol=TOL)


def test_attention_parity() -> None:
    q, k, v = np_project_qkv(H, W_Q, W_K, W_V)
    mask = np_causal_mask(5)
    np.testing.assert_allclose(
        np_attention_weights(q, k, mask=mask),
        tF.attention_weights(_t(q), _t(k), mask=_t(mask)).numpy(),
        atol=TOL,
    )
    np.testing.assert_allclose(
        np_sdpa(q, k, v, mask=mask),
        tF.scaled_dot_product_attention(_t(q), _t(k), _t(v), mask=_t(mask)).numpy(),
        atol=TOL,
    )


def test_multi_head_parity() -> None:
    mask = np_causal_mask(5)
    out_np, heads_np, _ = np_mha(H, W_Q, W_K, W_V, W_O, num_heads=2, mask=mask)
    out_t, heads_t = tF.multi_head_attention(
        _t(H), _t(W_Q), _t(W_K), _t(W_V), _t(W_O), num_heads=2, mask=_t(mask)
    )
    np.testing.assert_allclose(out_np, out_t.numpy(), atol=TOL)
    for expected, got in zip(heads_np, heads_t, strict=True):
        np.testing.assert_allclose(expected, got.numpy(), atol=TOL)


def test_cross_entropy_parity() -> None:
    logits = [0.0, -0.1725, 0.1725, -1.9539, 1.9539, 0.3907, -0.3907, 0.0]
    assert np_softmax_ce(logits, 5) == pytest.approx(
        float(tF.softmax_cross_entropy(_t(logits), 5)), abs=TOL
    )

    matrix = np.random.default_rng(0).normal(0, 1, (6, 8))
    targets = [1, 2, 3, 4, 5, 0]
    assert np_sequence_ce(matrix, targets) == pytest.approx(
        float(tF.sequence_cross_entropy(_t(matrix), _t(targets))), abs=TOL
    )
