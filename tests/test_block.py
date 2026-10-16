"""Uji blok Transformer (Bab 16) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.attention import causal_mask
from llmbepees.np.block import (
    attention_block_parameters,
    block_parameters,
    ffn_block_parameters,
    post_ln_block,
    pre_ln_block,
    stack_blocks,
)
from llmbepees.np.layernorm import layernorm
from llmbepees.np.multihead import multi_head_attention
from llmbepees.np.neuron import ffn

TOL = 1e-3

H0 = [
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
W_1 = 0.5 * np.array(
    [
        [1, 0, 1, -1, 0, 1, -1, 0],
        [0, 1, -1, 1, 1, 0, 0, -1],
        [1, 1, 0, 0, -1, 1, 0, 1],
        [-1, 0, 1, 1, 0, -1, 1, 0],
    ],
    dtype=float,
)
W_2 = 0.5 * W_1.T


def _mha(H: np.ndarray) -> np.ndarray:
    out, _, _ = multi_head_attention(H, W_Q, W_K, W_V, W_O, num_heads=2, mask=causal_mask(len(H)))
    return out


def _ffn(H: np.ndarray) -> np.ndarray:
    return ffn(H, W_1, None, W_2, None)


def test_pre_ln_block_with_zero_subblocks_is_identity() -> None:
    zero = lambda H: np.zeros_like(H)  # noqa: E731
    np.testing.assert_allclose(pre_ln_block(H0, zero, zero), H0, atol=1e-12)


def test_post_ln_block_with_zero_subblocks_normalizes_twice() -> None:
    zero = lambda H: np.zeros_like(H)  # noqa: E731
    expected = layernorm(layernorm(H0))
    np.testing.assert_allclose(post_ln_block(H0, zero, zero), expected, atol=1e-12)


def test_post_ln_block_mini_llm_final_state() -> None:
    out = post_ln_block(H0, _mha, _ffn)
    assert out.shape == (5, 4)
    assert out[4] == pytest.approx([-0.8678, -1.0861, 0.6953, 1.2585], abs=TOL)


def test_block_parameters_exact_and_convention() -> None:
    assert attention_block_parameters(4) == 64
    assert ffn_block_parameters(4, 8) == 64
    assert block_parameters(4, 8) == 144
    for d in (4, 512, 768):
        assert block_parameters(d, 4 * d) == 12 * d * d + 4 * d


def test_parameter_distribution_two_thirds_ffn() -> None:
    d = 768
    ffn_share = ffn_block_parameters(d, 4 * d) / block_parameters(d, 4 * d)
    assert ffn_share == pytest.approx(2 / 3, abs=0.01)


def test_stack_blocks_applies_in_order() -> None:
    inc = lambda H: H + 1.0  # noqa: E731
    out = stack_blocks(np.zeros((2, 2)), [inc, inc, inc])
    np.testing.assert_allclose(out, 3.0)


def test_pre_ln_and_post_ln_differ() -> None:
    pre = pre_ln_block(H0, _mha, _ffn)
    post = post_ln_block(H0, _mha, _ffn)
    assert not np.allclose(pre, post, atol=1e-2)
