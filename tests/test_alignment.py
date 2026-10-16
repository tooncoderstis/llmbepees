"""Uji penyelarasan pasca-pretraining: SFT, reward, RLHF, DPO, LoRA (Bab 20).

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.alignment import (
    bradley_terry,
    dpo_loss,
    lora_delta,
    lora_parameters,
    reward_model_loss,
    rlhf_objective,
    sft_loss,
)

TOL = 1e-3


def test_sft_loss_masks_question_tokens() -> None:
    log_probs = np.full(20, -1.0)
    mask = np.array([0] * 8 + [1] * 12)
    assert sft_loss(log_probs, mask) == pytest.approx(1.0)


def test_sft_loss_only_answer_values_matter() -> None:
    log_probs = np.array([-99.0] * 8 + [-1.0, -2.0, -3.0, -4.0] + [-99.0] * 8)
    mask = np.array([0] * 8 + [1] * 4 + [0] * 8)
    assert sft_loss(log_probs, mask) == pytest.approx((1 + 2 + 3 + 4) / 4)


def test_bradley_terry_and_reward_loss() -> None:
    assert bradley_terry(2.0, 0.5) == pytest.approx(0.8176, abs=TOL)
    assert reward_model_loss(2.0, 0.5) == pytest.approx(0.2014, abs=TOL)
    assert bradley_terry(0.5, 2.0) == pytest.approx(0.1824, abs=TOL)
    assert reward_model_loss(0.5, 2.0) == pytest.approx(1.7014, abs=TOL)


def test_reward_sensitivity_table() -> None:
    expected = [-3.0, -1.0, 0.0, 1.0, 3.0, 5.0]
    sigma = [0.0474, 0.2689, 0.5000, 0.7311, 0.9526, 0.9933]
    loss = [3.0486, 1.3133, 0.6931, 0.3133, 0.0486, 0.0067]
    for delta, s, ell in zip(expected, sigma, loss, strict=True):
        assert bradley_terry(delta, 0.0) == pytest.approx(s, abs=TOL)
        assert reward_model_loss(delta, 0.0) == pytest.approx(ell, abs=TOL)


def test_reward_shift_invariance() -> None:
    assert bradley_terry(2.0, 0.5) == pytest.approx(bradley_terry(102.0, 100.5))


def test_rlhf_objective_beta() -> None:
    assert rlhf_objective(2.0, 5.0, beta=0.01) == pytest.approx(1.95)
    assert rlhf_objective(2.0, 5.0, beta=0.40) == pytest.approx(0.0)
    assert rlhf_objective(2.0, 5.0, beta=1.00) == pytest.approx(-3.0)


def test_dpo_loss_example() -> None:
    loss = dpo_loss(-10.0, -10.5, -12.0, -11.5, beta=0.1)
    assert loss == pytest.approx(0.6444, abs=TOL)


def test_lora_delta_small_example() -> None:
    b = np.array([[0.1], [0.2], [0.0], [-0.1]])
    a = np.array([[1.0, 0.0, -0.5, 0.5]])
    delta = lora_delta(b, a, alpha=1.0, r=1)
    np.testing.assert_allclose(
        delta,
        [
            [0.10, 0.0, -0.05, 0.05],
            [0.20, 0.0, -0.10, 0.10],
            [0.00, 0.0, 0.00, 0.00],
            [-0.10, 0.0, 0.05, -0.05],
        ],
        atol=TOL,
    )


def test_lora_zero_init_is_identity() -> None:
    b = np.zeros((4, 1))
    a = np.array([[1.0, 0.0, -0.5, 0.5]])
    np.testing.assert_allclose(lora_delta(b, a, alpha=1.0, r=1), np.zeros((4, 4)))


def test_lora_parameter_savings() -> None:
    full = 4096 * 4096
    adapter = lora_parameters(4096, 4096, 8)
    assert adapter == 65536
    assert adapter / full == pytest.approx(0.0039, abs=1e-4)
