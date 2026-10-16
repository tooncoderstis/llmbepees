"""Uji pretraining: objektif, batch, scaling law (Bab 18) terhadap angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.training import (
    chinchilla_tokens,
    gradient_accumulation,
    random_guess_loss,
    scaling_law,
    sequence_cross_entropy,
    shift_pairs,
    tokens_per_step,
)

TOL = 1e-3


def test_shift_pairs() -> None:
    inputs, targets = shift_pairs([0, 1, 2, 3, 4, 5, 7])
    assert inputs.tolist() == [0, 1, 2, 3, 4, 5]
    assert targets.tolist() == [1, 2, 3, 4, 5, 7]


def test_sequence_cross_entropy_uniform_is_log_v() -> None:
    logits = np.zeros((3, 8))
    assert sequence_cross_entropy(logits, [0, 1, 2]) == pytest.approx(np.log(8), abs=TOL)


def test_sequence_cross_entropy_mean_over_positions() -> None:
    logits = np.zeros((2, 8))
    logits[0, 3] = 10.0  # posisi pertama sangat yakin ke token 3
    loss = sequence_cross_entropy(logits, [3, 0])
    assert loss == pytest.approx((0.0 + np.log(8)) / 2, abs=TOL)


def test_tokens_per_step() -> None:
    assert tokens_per_step(1562, 2048) == 1562 * 2048


def test_gradient_accumulation_is_mean() -> None:
    grads = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert gradient_accumulation(grads) == pytest.approx([3.0, 4.0])


def test_random_guess_loss() -> None:
    assert random_guess_loss(8) == pytest.approx(2.0794, abs=TOL)
    assert random_guess_loss(50257) == pytest.approx(10.82, abs=0.01)


def test_scaling_law_tenfold_improvement() -> None:
    factor = scaling_law(10.0, n_c=1.0, alpha=0.076)
    assert factor == pytest.approx(0.84, abs=0.01)


def test_chinchilla_tokens() -> None:
    assert chinchilla_tokens(1e9) == pytest.approx(2e10)
    assert chinchilla_tokens(7e9) == pytest.approx(1.4e11)
    assert chinchilla_tokens(70e9) == pytest.approx(1.4e12)
