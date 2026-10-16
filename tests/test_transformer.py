"""Uji model Transformer utuh & lapisan keluaran (Bab 17) terhadap angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.softmax import softmax
from llmbepees.np.transformer import (
    approx_model_parameters,
    decoder_forward,
    forward_flops,
    forward_macs,
    matrix_sizes,
    model_parameters,
    output_logits,
    predictions_per_sequence,
)

TOL = 1e-3

H_FINAL = [
    [1.4478, -0.6142, 0.3510, -1.1846],
    [0.7612, -0.3262, 1.0512, -1.4862],
    [-1.3887, 0.7940, -0.4977, 1.0924],
    [0.8409, 0.8070, -1.6252, -0.0227],
    [-0.8678, -1.0861, 0.6953, 1.2585],
]
E_MINI = [
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],
]


def test_output_logits_mini_llm_atas() -> None:
    Z = output_logits(H_FINAL, E_MINI)
    assert Z.shape == (5, 8)
    assert Z[4] == pytest.approx(
        [0.0, -0.1725, 0.1725, -1.9539, 1.9539, 0.3907, -0.3907, 0.0], abs=TOL
    )


def test_softmax_of_logits_is_distribution() -> None:
    Z = output_logits(H_FINAL, E_MINI)
    probs = softmax(Z[4])
    assert probs == pytest.approx(
        [0.0747, 0.0629, 0.0888, 0.0106, 0.5273, 0.1105, 0.0506, 0.0747], abs=TOL
    )
    assert float(np.sum(probs)) == pytest.approx(1.0)


def test_decoder_forward_matches_stack_then_output() -> None:
    def blk(H: np.ndarray) -> np.ndarray:
        return H

    Z = decoder_forward(H_FINAL, [blk], E_MINI)
    np.testing.assert_allclose(Z, output_logits(H_FINAL, E_MINI), atol=1e-12)


def test_decoder_forward_final_norm_changes_result() -> None:
    # H_FINAL sudah ternormalisasi (keluaran Post-LN), jadi pakai H mentah dulu.
    H_raw = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]

    def blk(H: np.ndarray) -> np.ndarray:
        return H

    a = decoder_forward(H_raw, [blk], E_MINI)
    b = decoder_forward(H_raw, [blk], E_MINI, final_norm=True)
    assert not np.allclose(a, b, atol=1e-2)


def test_approx_model_parameters_gpt2_and_gpt3() -> None:
    assert approx_model_parameters(50257, 768, 1024, 12) == 124_318_464
    assert approx_model_parameters(50257, 12288, 2048, 96) == 174_588_899_328


def test_model_parameters_includes_layer_norm_term() -> None:
    assert model_parameters(50257, 768, 1024, 12, 3072) == 124_355_328


def test_forward_macs_mini_llm() -> None:
    assert forward_macs(5, 4, 8, 8, num_heads=2) == 1000


def test_forward_flops_gpt3() -> None:
    assert forward_flops(2048, 175e9) == pytest.approx(7.168e14)


def test_predictions_per_sequence() -> None:
    assert predictions_per_sequence(7) == 6
    assert predictions_per_sequence(2048) == 2047
    assert predictions_per_sequence(1) == 0


def test_matrix_sizes_mini_llm() -> None:
    sizes = matrix_sizes(5, 4, 8, 8)
    assert sizes["embedding_X"] == "5 x 4"
    assert sizes["scores"] == "5 x 5"
    assert sizes["ffn_hidden"] == "5 x 8"
    assert sizes["logits"] == "5 x 8"
