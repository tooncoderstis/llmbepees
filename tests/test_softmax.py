"""Uji softmax, log-softmax, dan log-sum-exp (Bab 10) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.softmax import log_softmax, log_sum_exp, softmax

TOL = 1e-3

MINI_LOGITS = [0.0, -0.1725, 0.1725, -1.9539, 1.9539, 0.3907, -0.3907, 0.0]


def test_softmax_small_example() -> None:
    assert softmax([2.0, 1.0, 0.1]) == pytest.approx([0.6590, 0.2424, 0.0986], abs=TOL)


def test_softmax_mini_llm() -> None:
    p = softmax(MINI_LOGITS)
    assert p == pytest.approx(
        [0.0747, 0.0629, 0.0888, 0.0106, 0.5273, 0.1104, 0.0506, 0.0747], abs=TOL
    )
    assert float(np.sum(p)) == pytest.approx(1.0)


def test_softmax_is_positive_and_sums_to_one() -> None:
    p = softmax([3.0, 1.0, 0.5, -1.0])
    assert np.all(p > 0.0)
    assert float(np.sum(p)) == pytest.approx(1.0)


def test_softmax_shift_invariance() -> None:
    assert softmax([2.0, 1.0, 0.1]) == pytest.approx(softmax([1002.0, 1001.0, 1000.1]))


def test_softmax_numerically_stable_for_large_logits() -> None:
    p = softmax([1002.0, 1001.0, 1000.1])
    assert np.all(np.isfinite(p))
    assert p == pytest.approx([0.6590, 0.2424, 0.0986], abs=TOL)


def test_softmax_scale_sensitivity() -> None:
    z = [3.0, 1.0, 0.5, -1.0]
    assert softmax(z) == pytest.approx([0.8092, 0.1095, 0.0664, 0.0148], abs=TOL)
    assert softmax(np.asarray(z) * 8.0) == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=TOL)


def test_softmax_preserves_order_and_argmax() -> None:
    z = [3.0, 1.0, 0.5, -1.0]
    p = softmax(z)
    assert int(np.argmax(p)) == int(np.argmax(z)) == 0
    order = np.argsort(z)
    assert list(np.argsort(p)) == list(order)


def test_softmax_axis_for_batch() -> None:
    Z = [[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]]
    out = softmax(Z)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(np.sum(out, axis=-1), [1.0, 1.0], atol=TOL)
    np.testing.assert_allclose(out[1], [1 / 3, 1 / 3, 1 / 3], atol=TOL)


def test_log_sum_exp_matches_log_of_sum_of_exp() -> None:
    z = [2.0, 1.0, 0.1]
    assert log_sum_exp(z) == pytest.approx(np.log(np.sum(np.exp(z))), abs=TOL)


def test_log_sum_exp_mini_llm() -> None:
    assert log_sum_exp(MINI_LOGITS) == pytest.approx(2.5939, abs=TOL)


def test_log_sum_exp_is_stable_for_large_logits() -> None:
    val = log_sum_exp([1002.0, 1001.0, 1000.1])
    assert np.isfinite(val)
    assert val == pytest.approx(1002.0 + np.log(np.sum(np.exp([0.0, -1.0, -1.9]))), abs=TOL)


def test_log_softmax_equals_log_softmax() -> None:
    z = [2.0, 1.0, 0.1]
    np.testing.assert_allclose(log_softmax(z), np.log(softmax(z)), atol=1e-12)


def test_log_softmax_exponentiates_to_distribution() -> None:
    p = np.exp(log_softmax(MINI_LOGITS))
    np.testing.assert_allclose(p, softmax(MINI_LOGITS), atol=1e-12)
    assert float(np.sum(p)) == pytest.approx(1.0)
