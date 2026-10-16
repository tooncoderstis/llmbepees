"""Uji embedding (Bab 8) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.embedding import (
    embed_one_hot,
    embedding_lookup,
    embedding_parameters,
    one_hot,
    one_hot_matrix,
    tied_logits,
)
from llmbepees.np.vectors import cosine_similarity

TOL = 1e-3

E_MINI = [
    [0.0, 0.0, 0.0, 0.0],  # <bos> 0
    [1.0, 0.0, 1.0, 0.0],  # kucing 1
    [0.0, 1.0, 0.0, 1.0],  # duduk 2
    [1.0, 1.0, 0.0, 0.0],  # di 3
    [0.0, 0.0, 1.0, 1.0],  # atas 4
    [1.0, 0.0, 0.0, 1.0],  # meja 5
    [0.0, 1.0, 1.0, 0.0],  # tikar 6
    [1.0, 1.0, 1.0, 1.0],  # <eos> 7
]


def test_one_hot_examples() -> None:
    assert one_hot(0, 8) == pytest.approx([1, 0, 0, 0, 0, 0, 0, 0])
    assert one_hot(1, 8) == pytest.approx([0, 1, 0, 0, 0, 0, 0, 0])
    assert one_hot(5, 8) == pytest.approx([0, 0, 0, 0, 0, 1, 0, 0])


def test_one_hot_out_of_range() -> None:
    with pytest.raises(ValueError):
        one_hot(8, 8)


def test_one_hot_dot_properties() -> None:
    a = one_hot(3, 8)
    b = one_hot(5, 8)
    assert float(np.dot(a, b)) == pytest.approx(0.0)
    assert float(np.dot(a, a)) == pytest.approx(1.0)


def test_one_hot_matrix_rows() -> None:
    mat = one_hot_matrix([0, 1, 5], 8)
    assert mat.shape == (3, 8)
    np.testing.assert_allclose(mat[0], one_hot(0, 8))
    np.testing.assert_allclose(mat[2], one_hot(5, 8))
    assert mat.sum() == pytest.approx(3.0)


def test_embedding_lookup_meja() -> None:
    assert embedding_lookup(E_MINI, [5])[0] == pytest.approx([1.0, 0.0, 0.0, 1.0])


def test_embedding_lookup_sequence() -> None:
    ids = [0, 1, 2, 3, 4]
    np.testing.assert_allclose(embedding_lookup(E_MINI, ids), E_MINI[:5])


def test_embed_one_hot_equals_lookup() -> None:
    ids = [0, 1, 2, 3, 4, 5]
    one_hots = one_hot_matrix(ids, 8)
    np.testing.assert_allclose(embed_one_hot(E_MINI, one_hots), embedding_lookup(E_MINI, ids))


def test_tied_logits_meja_and_di() -> None:
    h = [-0.8678, -1.0861, 0.6953, 1.2585]
    logits = tied_logits(E_MINI, h)
    assert logits[5] == pytest.approx(0.3907, abs=TOL)
    assert logits[3] == pytest.approx(-1.9539, abs=TOL)


def test_embedding_parameter_counts() -> None:
    assert embedding_parameters(50257, 768) == 38_597_376
    assert embedding_parameters(32000, 4096) == 131_072_000
    assert embedding_parameters(50257, 12288) == 617_558_016


def test_cosine_meja_tikar_before_and_after_training() -> None:
    meja_init = [-0.0924, -0.3405, 0.6113, -0.0773]
    tikar_init = [-0.2142, -0.1761, 0.2662, 0.1827]
    assert cosine_similarity(meja_init, tikar_init) == pytest.approx(0.7557, abs=TOL)

    meja_trained = [-0.3744, -1.7296, 4.0288, -1.5429]
    tikar_trained = [-0.6374, -1.5958, 3.4239, -1.5035]
    assert cosine_similarity(meja_trained, tikar_trained) == pytest.approx(0.9960, abs=TOL)
