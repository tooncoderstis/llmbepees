"""Uji probabilitas (Bab 5) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.probability import (
    conditional_probability,
    cumulative_sum,
    expected_value,
    normal_pdf,
    sample_categorical,
    std,
    uniform,
    validate_distribution,
    variance,
)

TOL = 1e-3

P_SOFTMAX = [0.0001, 0.0, 0.0003, 0.0, 0.0005, 0.7492, 0.2494, 0.0006]


def test_validate_distribution_accepts_softmax_like() -> None:
    validate_distribution(P_SOFTMAX, atol=1e-3)


def test_validate_distribution_rejects_bad_sum() -> None:
    with pytest.raises(ValueError):
        validate_distribution([0.5, 0.4])


def test_validate_distribution_rejects_negative() -> None:
    with pytest.raises(ValueError):
        validate_distribution([1.2, -0.2])


def test_uniform() -> None:
    assert uniform(8) == pytest.approx([0.125] * 8)


def test_conditional_probability_meja_tikar() -> None:
    assert conditional_probability(0.30, 0.40) == pytest.approx(0.75, abs=TOL)
    assert conditional_probability(0.08, 0.40) == pytest.approx(0.20, abs=TOL)


def test_conditional_probability_rejects_zero() -> None:
    with pytest.raises(ValueError):
        conditional_probability(0.1, 0.0)


def test_expected_value_dice() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    probs = [0.1, 0.2, 0.3, 0.4]
    assert expected_value(values, probs) == pytest.approx(3.0, abs=TOL)


def test_expected_value_need_not_be_possible_value() -> None:
    assert expected_value([1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.0, 0.0]) == pytest.approx(1.5)


def test_variance_and_std() -> None:
    h = [2.0, 4.0, 4.0, 6.0]
    probs = [0.25, 0.25, 0.25, 0.25]
    assert variance(h, probs) == pytest.approx(2.0, abs=TOL)
    assert std(h, probs) == pytest.approx(1.4142, abs=TOL)


def test_layernorm_normalization() -> None:
    h = [2.0, 4.0, 4.0, 6.0]
    probs = [0.25, 0.25, 0.25, 0.25]
    mean = expected_value(h, probs)
    sigma = std(h, probs)
    h_hat = (np.asarray(h) - mean) / sigma
    np.testing.assert_allclose(h_hat, [-1.4142, 0.0, 0.0, 1.4142], atol=TOL)
    assert expected_value(h_hat, probs) == pytest.approx(0.0, abs=TOL)
    assert variance(h_hat, probs) == pytest.approx(1.0, abs=TOL)


def test_cumulative_sum() -> None:
    assert cumulative_sum([0.5, 0.3, 0.15, 0.05]) == pytest.approx([0.5, 0.8, 0.95, 1.0])


def test_normal_pdf_peak() -> None:
    assert normal_pdf([0.0])[0] == pytest.approx(0.3989, abs=TOL)


def test_normal_pdf_is_symmetric() -> None:
    assert normal_pdf([-1.0])[0] == pytest.approx(normal_pdf([1.0])[0], abs=TOL)


def test_sample_categorical_deterministic() -> None:
    p = [0.5, 0.3, 0.15, 0.05]
    assert sample_categorical(p, u=0.0) == 0
    assert sample_categorical(p, u=0.62) == 1
    assert sample_categorical(p, u=0.99) == 3


def test_sample_categorical_boundary_belongs_to_bucket() -> None:
    assert sample_categorical([0.5, 0.3, 0.15, 0.05], u=0.5) == 1


def test_sample_categorical_empirical() -> None:
    p = [0.5, 0.3, 0.15, 0.05]
    rng = np.random.default_rng(42)
    draws = [sample_categorical(p, rng=rng) for _ in range(20_000)]
    freq = np.bincount(draws, minlength=4) / len(draws)
    np.testing.assert_allclose(freq, p, atol=0.02)


def test_sample_categorical_rejects_invalid_u() -> None:
    with pytest.raises(ValueError):
        sample_categorical([0.5, 0.5], u=1.0)
