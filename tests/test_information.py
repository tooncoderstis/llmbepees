"""Uji teori informasi (Bab 6) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002). Basis ``e`` = nat, basis ``2`` = bit.
"""

import math

import pytest

from llmbepees.np.information import (
    bits_per_byte,
    cross_entropy,
    entropy,
    information,
    kl_divergence,
    mean_negative_log_likelihood,
    perplexity,
)

TOL = 1e-3

P = [0.5, 0.25, 0.125, 0.125]
Q_UNIFORM4 = [0.25, 0.25, 0.25, 0.25]


def test_information_nat_and_bit() -> None:
    assert information(0.5) == pytest.approx(0.6931, abs=TOL)
    assert information(0.5, base=2.0) == pytest.approx(1.0, abs=TOL)
    assert information(1.0) == pytest.approx(0.0, abs=TOL)


def test_information_table_extremes() -> None:
    assert information(0.001) == pytest.approx(6.908, abs=TOL)
    assert information(0.001, base=2.0) == pytest.approx(9.966, abs=TOL)


def test_information_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        information(0.0)
    with pytest.raises(ValueError):
        information(1.5)


def test_entropy_nonuniform_bits() -> None:
    assert entropy(P, base=2.0) == pytest.approx(1.750, abs=TOL)


def test_entropy_uniform_bits() -> None:
    assert entropy(Q_UNIFORM4, base=2.0) == pytest.approx(2.000, abs=TOL)


def test_entropy_max_is_log_n() -> None:
    assert entropy([0.125] * 8, base=2.0) == pytest.approx(3.0, abs=TOL)
    assert entropy([0.125] * 8) == pytest.approx(2.0794, abs=TOL)


def test_entropy_min_is_zero() -> None:
    assert entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=TOL)


def test_cross_entropy_example() -> None:
    assert cross_entropy(P, Q_UNIFORM4, base=2.0) == pytest.approx(2.000, abs=TOL)


def test_cross_entropy_geq_entropy() -> None:
    h_p = entropy(P, base=2.0)
    h_pq = cross_entropy(P, Q_UNIFORM4, base=2.0)
    assert h_pq >= h_p - TOL
    assert h_pq - h_p == pytest.approx(0.250, abs=TOL)


def test_cross_entropy_rejects_zero_where_needed() -> None:
    with pytest.raises(ValueError):
        cross_entropy([1.0, 0.0], [0.0, 1.0])


def test_kl_divergence_matches_difference() -> None:
    assert kl_divergence(P, Q_UNIFORM4, base=2.0) == pytest.approx(0.250, abs=TOL)


def test_kl_divergence_zero_when_equal() -> None:
    assert kl_divergence(P, P, base=2.0) == pytest.approx(0.0, abs=TOL)


def test_kl_divergence_nonnegative_and_asymmetric() -> None:
    p = [0.9, 0.1]
    q = [0.5, 0.5]
    assert kl_divergence(p, q) >= 0.0
    assert kl_divergence(q, p) >= 0.0
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p), abs=TOL)


def test_loss_model_demonstration_vs_trained() -> None:
    assert -math.log(0.1105) == pytest.approx(2.2032, abs=TOL)
    assert -math.log(0.7492) == pytest.approx(0.2887, abs=TOL)


def test_mean_negative_log_likelihood() -> None:
    probs = [0.5, 0.25, 0.4, 0.1]
    assert mean_negative_log_likelihood(probs) == pytest.approx(1.3246, abs=TOL)


def test_perplexity_from_loss() -> None:
    assert perplexity(1.3246) == pytest.approx(3.7606, abs=TOL)
    assert perplexity(0.0) == pytest.approx(1.0, abs=TOL)
    assert perplexity(math.log(8.0)) == pytest.approx(8.0, abs=TOL)


def test_perplexity_training_checkpoints() -> None:
    assert perplexity(2.4480) == pytest.approx(11.57, abs=0.01)
    assert perplexity(0.0955) == pytest.approx(1.10, abs=0.01)


def test_bits_per_byte() -> None:
    assert bits_per_byte(375.0, 1000) == pytest.approx(0.541, abs=TOL)
