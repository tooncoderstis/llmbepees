"""Uji loss softmax + cross-entropy, gradien, MSE, dan label smoothing (Bab 10).

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.calculus import numerical_gradient
from llmbepees.np.embedding import one_hot
from llmbepees.np.information import cross_entropy
from llmbepees.np.loss import (
    label_smoothing,
    softmax_cross_entropy,
    softmax_cross_entropy_gradient,
    softmax_mse_gradient,
    softmax_mse_loss,
)
from llmbepees.np.softmax import softmax

TOL = 1e-3

MINI_LOGITS = [0.0, -0.1725, 0.1725, -1.9539, 1.9539, 0.3907, -0.3907, 0.0]
MEJA = 5


def test_softmax_cross_entropy_mini_llm() -> None:
    assert softmax_cross_entropy(MINI_LOGITS, MEJA) == pytest.approx(2.2032, abs=TOL)


def test_softmax_cross_entropy_equals_negative_log_probability() -> None:
    logits = [2.0, 1.0, 0.1]
    p = softmax(logits)
    for c in range(3):
        assert softmax_cross_entropy(logits, c) == pytest.approx(-np.log(p[c]), abs=1e-12)


def test_parity_with_information_cross_entropy() -> None:
    p = softmax(MINI_LOGITS)
    y = one_hot(MEJA, len(MINI_LOGITS))
    assert softmax_cross_entropy(MINI_LOGITS, MEJA) == pytest.approx(cross_entropy(y, p), abs=1e-12)


def test_softmax_cross_entropy_gradient_is_p_minus_y() -> None:
    grad = softmax_cross_entropy_gradient([2.0, 1.0, 0.1], 0)
    assert grad == pytest.approx([-0.3410, 0.2424, 0.0986], abs=TOL)


def test_softmax_cross_entropy_gradient_matches_numerical() -> None:
    logits = [2.0, 1.0, 0.1]

    def loss(z: np.ndarray) -> float:
        return softmax_cross_entropy(z, 0)

    grad = softmax_cross_entropy_gradient(logits, 0)
    np.testing.assert_allclose(grad, numerical_gradient(loss, logits), atol=1e-4)


def test_softmax_cross_entropy_gradient_sums_to_zero() -> None:
    grad = softmax_cross_entropy_gradient(MINI_LOGITS, MEJA)
    assert float(np.sum(grad)) == pytest.approx(0.0, abs=1e-9)


def test_mini_llm_gradient_table() -> None:
    grad = softmax_cross_entropy_gradient(MINI_LOGITS, MEJA)
    assert grad == pytest.approx(
        [0.0747, 0.0629, 0.0888, 0.0106, 0.5273, -0.8895, 0.0506, 0.0747], abs=TOL
    )


def test_label_smoothing_example() -> None:
    y = label_smoothing(MEJA, 8, 0.1)
    assert y[MEJA] == pytest.approx(0.9125, abs=TOL)
    assert y[0] == pytest.approx(0.0125, abs=TOL)
    assert float(np.sum(y)) == pytest.approx(1.0)


def test_label_smoothing_zero_epsilon_is_one_hot() -> None:
    np.testing.assert_allclose(label_smoothing(MEJA, 8, 0.0), one_hot(MEJA, 8))


def test_label_smoothing_invalid_epsilon() -> None:
    with pytest.raises(ValueError):
        label_smoothing(MEJA, 8, 1.5)


def test_softmax_mse_gradient_matches_numerical() -> None:
    logits = [2.0, 1.0, 0.1]

    def loss(z: np.ndarray) -> float:
        return softmax_mse_loss(z, 0)

    grad = softmax_mse_gradient(logits, 0)
    np.testing.assert_allclose(grad, numerical_gradient(loss, logits), atol=1e-4)


def _distribution_with_correct_class(p_c: float, num_classes: int) -> np.ndarray:
    p = np.full(num_classes, (1.0 - p_c) / (num_classes - 1))
    p[0] = p_c
    return p


def test_mse_gradient_vanishes_when_model_very_wrong() -> None:
    num_classes = 5
    for p_c, expected_mse in [(0.001, -0.0025), (0.5, -0.3125), (0.999, -0.0000025)]:
        p = _distribution_with_correct_class(p_c, num_classes)
        logits = np.log(p)  # softmax(log p) == p
        ce_grad = float(softmax_cross_entropy_gradient(logits, 0)[0])
        mse_grad = float(softmax_mse_gradient(logits, 0)[0])
        assert ce_grad == pytest.approx(p_c - 1.0, abs=1e-6)
        assert mse_grad == pytest.approx(expected_mse, abs=TOL)


def test_cross_entropy_gradient_far_larger_than_mse_when_wrong() -> None:
    p = _distribution_with_correct_class(0.001, 5)
    logits = np.log(p)
    ce_grad = abs(float(softmax_cross_entropy_gradient(logits, 0)[0]))
    mse_grad = abs(float(softmax_mse_gradient(logits, 0)[0]))
    assert ce_grad / mse_grad > 100.0
