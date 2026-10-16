"""Uji operasi matriks (Bab 4) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002); perbandingan matriks memakai
``numpy.testing.assert_allclose``.
"""

import numpy as np
import pytest

from llmbepees.np.matrices import (
    as_matrix,
    flops_matmul,
    identity,
    matadd,
    matmul,
    matvec,
    transpose,
)

TOL = 1e-3

A_3x2 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
B_2x3 = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]

X_TOKENS = [
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
]

P_POSISI = [
    [0.5, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.5],
    [0.5, 0.5, 0.0, 0.0],
]

H0 = [
    [0.5, 0.0, 0.0, 0.0],
    [1.0, 0.5, 1.0, 0.0],
    [0.0, 1.0, 0.5, 1.0],
    [1.0, 1.0, 0.0, 0.5],
    [0.5, 0.5, 1.0, 1.0],
]

W_Q = [
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
]

Q_EXPECTED = [
    [0.5, 0.5, 0.5, 0.5],
    [1.5, 1.5, 1.5, 2.0],
    [2.0, 1.0, 1.0, 1.5],
    [2.5, 2.0, 2.0, 1.5],
    [2.0, 1.0, 1.0, 2.5],
]


def test_as_matrix_rejects_vector() -> None:
    with pytest.raises(ValueError):
        as_matrix([1.0, 2.0])


def test_transpose() -> None:
    np.testing.assert_allclose(transpose(A_3x2), [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])


def test_matadd_embedding_plus_position() -> None:
    np.testing.assert_allclose(matadd(X_TOKENS, P_POSISI), H0, atol=TOL)


def test_matadd_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        matadd(A_3x2, B_2x3)


def test_matmul_full_example() -> None:
    expected = [[27.0, 30.0, 33.0], [61.0, 68.0, 75.0], [95.0, 106.0, 117.0]]
    np.testing.assert_allclose(matmul(A_3x2, B_2x3), expected, atol=TOL)


def test_matmul_middle_dimension_must_match() -> None:
    with pytest.raises(ValueError):
        matmul(A_3x2, A_3x2)


def test_matmul_transpose_identity() -> None:
    product = matmul(A_3x2, B_2x3)
    np.testing.assert_allclose(
        transpose(product), matmul(transpose(B_2x3), transpose(A_3x2)), atol=TOL
    )


def test_matvec_query_for_atas() -> None:
    h_atas = [0.5, 0.5, 1.0, 1.0]
    assert matvec(h_atas, W_Q) == pytest.approx([2.0, 1.0, 1.0, 2.5], abs=TOL)


def test_matmul_batch_all_tokens() -> None:
    np.testing.assert_allclose(matmul(H0, W_Q), Q_EXPECTED, atol=TOL)


def test_identity_is_multiplicative_unit() -> None:
    np.testing.assert_allclose(matmul(A_3x2, identity(2)), A_3x2, atol=TOL)


def test_flops_gpt3_query_projection() -> None:
    assert flops_matmul(2048, 12288, 12288) == 618_475_290_624


def test_returns_float64() -> None:
    assert np.asarray(matmul(A_3x2, B_2x3)).dtype == np.float64
