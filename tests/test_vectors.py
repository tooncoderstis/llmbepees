"""Uji operasi vektor (Bab 3) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002): semua nilai dibandingkan dengan ``abs=1e-3``.
"""

import numpy as np
import pytest

from llmbepees.np.vectors import (
    add,
    as_vector,
    cosine_similarity,
    dot,
    hadamard,
    norm_l2,
    scale,
)

TOL = 1e-3


def test_as_vector_rejects_matrix() -> None:
    with pytest.raises(ValueError):
        as_vector([[1.0, 2.0]])


def test_add_embedding_plus_position() -> None:
    embedding = [1.0, 0.0, 1.0, 0.0]
    position = [0.0, 0.5, 0.0, 0.0]
    assert add(embedding, position) == pytest.approx([1.0, 0.5, 1.0, 0.0])


def test_add_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        add([1.0, 2.0], [1.0, 2.0, 3.0])


def test_scale() -> None:
    assert scale(0.5, [2.0, -4.0, 1.0, 6.0]) == pytest.approx([1.0, -2.0, 0.5, 3.0])


def test_dot_first_example() -> None:
    assert dot([2.0, 3.0], [4.0, -1.0]) == pytest.approx(5.0, abs=TOL)


def test_dot_attention_score() -> None:
    q = [2.0, 1.0, 1.0, 2.5]
    k = [1.5, 1.0, 1.5, 1.0]
    assert dot(q, k) == pytest.approx(8.0, abs=TOL)


def test_dot_is_symmetric() -> None:
    a = [1.5, -2.0, 0.25]
    b = [2.0, 0.5, 4.0]
    assert dot(a, b) == pytest.approx(dot(b, a), abs=TOL)


def test_norm_l2() -> None:
    assert norm_l2([2.0, 3.0]) == pytest.approx(3.6056, abs=TOL)
    assert norm_l2([4.0, -1.0]) == pytest.approx(4.1231, abs=TOL)


def test_norm_squared_equals_self_dot() -> None:
    a = [2.0, 3.0]
    assert norm_l2(a) ** 2 == pytest.approx(dot(a, a), abs=TOL)


def test_norm_self_check() -> None:
    assert norm_l2([1.0, 2.0, 2.0]) == pytest.approx(3.0, abs=TOL)


def test_cosine_first_example() -> None:
    assert cosine_similarity([2.0, 3.0], [4.0, -1.0]) == pytest.approx(0.3363, abs=TOL)


def test_cosine_raja_ratu_vs_traktor() -> None:
    raja = [0.9, 0.1, 0.4]
    ratu = [0.8, 0.2, 0.5]
    traktor = [-0.7, 0.6, -0.2]
    assert cosine_similarity(raja, ratu) == pytest.approx(0.9846, abs=TOL)
    assert cosine_similarity(raja, traktor) == pytest.approx(-0.6960, abs=TOL)


def test_cosine_self_check() -> None:
    assert cosine_similarity([1.0, 2.0, 2.0], [2.0, 0.0, 1.0]) == pytest.approx(0.5963, abs=TOL)


def test_cosine_orthogonal_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=TOL)


def test_cosine_rejects_zero_vector() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 0.0], [1.0, 1.0])


def test_hadamard_is_not_dot() -> None:
    a = [2.0, 3.0]
    b = [4.0, -1.0]
    assert hadamard(a, b) == pytest.approx([8.0, -3.0])
    assert dot(a, b) == pytest.approx(5.0, abs=TOL)


def test_returns_float64_arrays() -> None:
    assert np.asarray(add([1, 2], [3, 4])).dtype == np.float64
