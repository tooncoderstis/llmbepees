"""Uji mini-LLM 188 parameter (Bab 21) — reproduksi eksperimen dari nol.

Toleransi desimal ke-3 (ADR-0002). Latih sekali (module scope) lalu periksa
seluruh titik pemeriksaan buku.
"""

import numpy as np
import pytest

from llmbepees.np.mini_llm import (
    SENTENCE_MEJA,
    TRAIN_SEQUENCES,
    VOCAB,
    finite_difference_gradients,
    forward,
    generate,
    init_params,
    next_token_probs,
    params_count,
    sequence_loss,
    train,
)

TOL = 1e-3

BOOK_HISTORY = {
    1: 2.1266,
    2: 1.9002,
    5: 1.4479,
    10: 1.0009,
    20: 0.6009,
    50: 0.1770,
    100: 0.1100,
    200: 0.0984,
    300: 0.0964,
    400: 0.0955,
}


@pytest.fixture(scope="module")
def trained():
    return train()


def test_parameter_count_is_188() -> None:
    params = init_params()
    assert params_count(params) == 188
    assert params["P"].shape == (7, 4)
    assert params["E"].shape == (8, 4)


def test_initial_embedding_matches_book() -> None:
    params = init_params(42)
    assert params["E"][5] == pytest.approx([-0.0924, -0.3405, 0.6113, -0.0773], abs=TOL)
    assert params["E"][6] == pytest.approx([-0.2142, -0.1761, 0.2662, 0.1827], abs=TOL)


def test_initial_loss_is_2_4480() -> None:
    params = init_params()
    assert sequence_loss(params, TRAIN_SEQUENCES) == pytest.approx(2.4480, abs=TOL)


def test_initial_distribution_matches_book() -> None:
    params = init_params()
    probs = next_token_probs(params, SENTENCE_MEJA[:5])
    assert probs == pytest.approx(
        [0.2373, 0.0975, 0.1858, 0.0182, 0.1736, 0.1298, 0.0862, 0.0715], abs=TOL
    )


def test_forward_shapes() -> None:
    params = init_params()
    assert forward(params, SENTENCE_MEJA).shape == (7, VOCAB)
    assert forward(params, TRAIN_SEQUENCES).shape == (4, 7, VOCAB)


def test_finite_difference_gradient_matches() -> None:
    params = init_params()
    grads = finite_difference_gradients(params, TRAIN_SEQUENCES)
    norm = float(np.sqrt(sum(np.sum(g * g) for g in grads.values())))
    largest = max(float(np.max(np.abs(g))) for g in grads.values())
    assert norm == pytest.approx(1.0442, abs=1e-3)
    assert largest == pytest.approx(0.4703, abs=1e-3)


def test_loss_trajectory_matches_book(trained) -> None:
    _, history = trained
    for step, expected in BOOK_HISTORY.items():
        assert history[step] == pytest.approx(expected, abs=TOL)


def test_final_loss_and_perplexity(trained) -> None:
    params, _ = trained
    loss = sequence_loss(params, TRAIN_SEQUENCES)
    assert loss == pytest.approx(0.0955, abs=TOL)
    assert float(np.exp(loss)) == pytest.approx(1.10, abs=0.01)


def test_final_distribution_matches_data_proportions(trained) -> None:
    params, _ = trained
    probs = next_token_probs(params, SENTENCE_MEJA[:5])
    assert probs[5] == pytest.approx(0.7492, abs=TOL)  # meja
    assert probs[6] == pytest.approx(0.2494, abs=TOL)  # tikar


def test_theoretical_minimum_loss() -> None:
    minimum = (3 * (-np.log(0.75)) + 1 * (-np.log(0.25))) / 24
    assert minimum == pytest.approx(0.0937, abs=TOL)


def test_meja_tikar_embeddings_become_parallel(trained) -> None:
    params, _ = trained
    meja, tikar = params["E"][5], params["E"][6]
    cosine = float(meja @ tikar / (np.linalg.norm(meja) * np.linalg.norm(tikar)))
    assert cosine == pytest.approx(0.9960, abs=TOL)


def test_last_positional_row_is_untouched(trained) -> None:
    params, _ = trained
    assert np.allclose(params["P"][6], init_params()["P"][6])


def test_generate_completes_sentence(trained) -> None:
    params, _ = trained
    assert generate(params, SENTENCE_MEJA[:5]) == [0, 1, 2, 3, 4, 5, 7]


def test_attention_becomes_one_hot(trained) -> None:
    params, _ = trained
    x = np.asarray(SENTENCE_MEJA, dtype=np.int64)
    h = params["E"][x] + params["P"][:7]
    q = h @ params["Wq"]
    k = h @ params["Wk"]
    scores = q @ k.T / np.sqrt(4)
    scores = scores + np.where(np.tril(np.ones((7, 7), bool)), 0.0, -1e9)
    scores = scores - scores.max(axis=-1, keepdims=True)
    attention = np.exp(scores)
    attention = attention / attention.sum(axis=-1, keepdims=True)
    # baris terakhir (setelah "di atas") memusatkan hampir seluruh bobot ke diri sendiri
    assert attention[4, 4] == pytest.approx(0.9998, abs=TOL)


def test_initial_attention_is_nearly_uniform() -> None:
    params = init_params()
    x = np.asarray(SENTENCE_MEJA, dtype=np.int64)
    h = params["E"][x] + params["P"][:7]
    q = h @ params["Wq"]
    k = h @ params["Wk"]
    scores = q @ k.T / np.sqrt(4)
    scores = scores + np.where(np.tril(np.ones((7, 7), bool)), 0.0, -1e9)
    scores = scores - scores.max(axis=-1, keepdims=True)
    attention = np.exp(scores)
    attention = attention / attention.sum(axis=-1, keepdims=True)
    assert float(np.max(attention[4])) < 0.3
