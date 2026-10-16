"""Uji inference: sampling, temperature, top-k/p, beam, KV cache (Bab 19).

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.inference import (
    beam_step,
    decode_next,
    decode_speed_upper_bound,
    greedy,
    prefill_work,
    sample,
    temperature_softmax,
    top_k,
    top_p,
)

TOL = 1e-3

DEMO_LOGITS = [0.0, -0.1725, 0.1725, -1.9539, 1.9539, 0.3907, -0.3907, 0.0]
DEMO_PROBS = [0.0747, 0.0629, 0.0888, 0.0106, 0.5273, 0.1105, 0.0506, 0.0747]


def test_greedy_picks_argmax() -> None:
    assert greedy(DEMO_LOGITS) == 4
    assert greedy([0.7492, 0.2494]) == 0


def test_temperature_table() -> None:
    assert temperature_softmax(DEMO_LOGITS, 0.5) == pytest.approx(
        [0.0177, 0.0125, 0.0250, 0.0004, 0.8801, 0.0386, 0.0081, 0.0177], abs=TOL
    )
    assert temperature_softmax(DEMO_LOGITS, 2.0) == pytest.approx(
        [0.1102, 0.1010, 0.1201, 0.0415, 0.2926, 0.1339, 0.0906, 0.1102], abs=TOL
    )


def test_temperature_approaches_greedy() -> None:
    probs = temperature_softmax(DEMO_LOGITS, 0.1)
    assert probs == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], abs=1e-4)


def test_top_k_example() -> None:
    out = top_k(DEMO_PROBS, 3)
    assert out[4] == pytest.approx(0.7257, abs=TOL)
    assert out[5] == pytest.approx(0.1520, abs=TOL)
    assert out[2] == pytest.approx(0.1222, abs=TOL)
    assert float(np.sum(out)) == pytest.approx(1.0)


def test_top_p_example() -> None:
    out = top_p(DEMO_PROBS, 0.8)
    nonzero = np.flatnonzero(out)
    assert nonzero.tolist() == [0, 2, 4, 5]  # <bos>, duduk, atas, meja
    assert out[4] == pytest.approx(0.6581, abs=TOL)
    assert out[5] == pytest.approx(0.1379, abs=TOL)
    assert out[2] == pytest.approx(0.1108, abs=TOL)
    assert out[0] == pytest.approx(0.0932, abs=TOL)


def test_top_p_adaptive_on_sharp_distribution() -> None:
    trained = [0.0001, 0.0, 0.0003, 0.0, 0.0005, 0.7492, 0.2494, 0.0006]
    assert int(np.count_nonzero(top_p(trained, 0.8))) == 2


def test_sample_is_reproducible() -> None:
    probs = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    rng = np.random.default_rng(0)
    assert sample(probs, rng) == 4


def test_decode_next_pipeline() -> None:
    rng = np.random.default_rng(1)
    token, probs = decode_next(DEMO_LOGITS, temperature=1.0, k=3, p=1.0, rng=rng)
    assert token in (2, 4, 5)
    assert float(np.sum(probs)) == pytest.approx(1.0)


def test_beam_step_two_steps() -> None:
    step1 = beam_step([((), 0.0)], [[-0.693, -1.204, -np.inf, -np.inf]], width=2)
    assert [(seq, round(s, 3)) for seq, s in step1] == [((0,), -0.693), ((1,), -1.204)]

    log_probs = [[-np.inf, -np.inf, -0.357, -1.609], [-np.inf, -np.inf, -0.223, -1.204]]
    step2 = beam_step(step1, log_probs, width=2)
    assert [(seq, round(s, 3)) for seq, s in step2] == [((0, 2), -1.050), ((1, 2), -1.427)]


def test_prefill_work_cache_saves_recompute() -> None:
    assert prefill_work(5, 100, use_cache=False) == 5450
    assert prefill_work(5, 100, use_cache=True) == 105


def test_decode_speed_upper_bound() -> None:
    assert decode_speed_upper_bound(2000.0, 14.0) == pytest.approx(142.857, abs=0.01)
