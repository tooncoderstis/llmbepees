"""Uji optimizer (Bab 12) terhadap contoh angka buku.

Toleransi desimal ke-3 (ADR-0002).
"""

import numpy as np
import pytest

from llmbepees.np.calculus import gradient_descent_step
from llmbepees.np.optimizer import (
    adam_memory_bytes,
    adam_step,
    adamw_step,
    gradient_clip,
    lr_cosine,
    lr_warmup,
    mini_batch_gradient,
    momentum_limit,
    momentum_step,
    noise_std_after_batch,
    sgd_step,
)

TOL = 1e-3


def _sgd_sequence(lr: float, steps: int) -> list[float]:
    w = np.array([0.0])
    seq = [0.0]
    for _ in range(steps - 1):
        w = sgd_step(w, np.array([2.0 * (w[0] - 3.0)]), lr=lr)
        seq.append(round(float(w[0]), 4))
    return seq


def test_sgd_sequence_converges() -> None:
    assert _sgd_sequence(0.1, 6) == pytest.approx([0.0, 0.6, 1.08, 1.464, 1.7712, 2.0170])


def test_sgd_sequence_diverges_with_large_lr() -> None:
    assert _sgd_sequence(1.1, 5) == pytest.approx([0.0, 6.6, -1.32, 8.184, -3.2208])


def test_sgd_step_matches_gradient_descent_step() -> None:
    assert sgd_step([1.0, -2.0], [0.5, 0.25], lr=0.1) == pytest.approx(
        gradient_descent_step([1.0, -2.0], [0.5, 0.25], lr=0.1)
    )


def test_stability_bound_for_quadratic() -> None:
    lr = 0.9  # η < 1/a, a = 1 → stabil (faktor = 1 − 2η = −0,8)
    w = np.array([0.0])
    for _ in range(200):
        w = sgd_step(w, np.array([2.0 * (w[0] - 3.0)]), lr=lr)
    assert float(w[0]) == pytest.approx(3.0, abs=1e-3)


def test_mini_batch_gradient_is_mean() -> None:
    assert mini_batch_gradient([[1.0, 2.0], [3.0, 4.0]]) == pytest.approx([2.0, 3.0])


def test_noise_std_shrinks_with_batch() -> None:
    assert noise_std_after_batch(0.5, 100, 10_000) == pytest.approx(0.05, abs=1e-9)


def test_momentum_oscillating_gradients() -> None:
    velocity = np.array([0.0])
    seq = []
    for g in [1.0, -0.8, 0.9, -0.7]:
        _, velocity = momentum_step([0.0], [g], velocity, lr=0.1, beta=0.9)
        seq.append(round(float(velocity[0]), 3))
    assert seq == pytest.approx([1.0, 0.1, 0.99, 0.191])


def test_momentum_consistent_gradients() -> None:
    velocity = np.array([0.0])
    seq = []
    for _ in range(4):
        _, velocity = momentum_step([0.0], [0.3], velocity, lr=0.1, beta=0.9)
        seq.append(round(float(velocity[0]), 3))
    assert seq == pytest.approx([0.3, 0.57, 0.813, 1.032])


def test_momentum_limit_speeds_up_tenfold() -> None:
    assert momentum_limit(0.3, beta=0.9) == pytest.approx(3.0)
    assert momentum_limit(1.0, beta=0.9) == pytest.approx(10.0)


def _adam_sequence(lr: float, steps: int) -> list[float]:
    w = np.array([0.0])
    m = np.array([0.0])
    v = np.array([0.0])
    seq: list[float] = []
    for t in range(1, steps + 1):
        g = np.array([2.0 * (w[0] - 3.0)])
        w, m, v = adam_step(w, g, m, v, t=t, lr=lr)
        seq.append(round(float(w[0]), 4))
    return seq


def test_adam_sequence_step_size_is_learning_rate() -> None:
    assert _adam_sequence(0.1, 5) == pytest.approx([0.1, 0.1999, 0.2996, 0.3991, 0.4982])


def test_adam_first_step_bias_correction() -> None:
    zero = np.array([0.0])
    w, m, v = adam_step(zero, np.array([-6.0]), zero, zero, t=1, lr=0.1)
    assert float(m[0]) == pytest.approx(-0.6)
    assert float(v[0]) == pytest.approx(0.036)
    assert float(w[0]) == pytest.approx(0.1)  # langkah ≈ η (m̂/√v̂ ≈ −1)


def test_adam_matches_sgd_vs_adam_comparison() -> None:
    sgd = _sgd_sequence(0.1, 6)[1:]
    adam = _adam_sequence(0.1, 5)
    assert sgd == pytest.approx([0.6, 1.08, 1.464, 1.7712, 2.0170])
    assert adam == pytest.approx([0.1, 0.1999, 0.2996, 0.3991, 0.4982])


def test_adamw_applies_weight_decay_separately() -> None:
    new_p, _, _ = adamw_step(
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        t=1,
        lr=0.1,
        weight_decay=0.1,
    )
    assert float(new_p[0]) == pytest.approx(1.0 - 0.1 * 0.1 * 1.0)


def test_gradient_clip_example() -> None:
    assert gradient_clip([3.0, 4.0], max_norm=1.0) == pytest.approx([0.6, 0.8])
    assert float(np.linalg.norm(gradient_clip([3.0, 4.0], max_norm=1.0))) == pytest.approx(1.0)


def test_gradient_clip_leaves_small_gradient_unchanged() -> None:
    assert gradient_clip([0.1, 0.2], max_norm=1.0) == pytest.approx([0.1, 0.2])


def test_lr_warmup_schedule() -> None:
    kwargs = {"max_lr": 6e-4, "warmup_steps": 2000}
    assert lr_warmup(0, **kwargs) == pytest.approx(0.0)
    assert lr_warmup(500, **kwargs) == pytest.approx(1.5e-4)
    assert lr_warmup(1000, **kwargs) == pytest.approx(3.0e-4)
    assert lr_warmup(2000, **kwargs) == pytest.approx(6.0e-4)


def test_lr_cosine_schedule() -> None:
    kwargs = {"max_lr": 6e-4, "min_lr": 6e-5, "warmup_steps": 2000, "total_steps": 100_000}
    assert lr_cosine(2000, **kwargs) == pytest.approx(6.0e-4)
    assert lr_cosine(26_500, **kwargs) == pytest.approx(5.2e-4, rel=0.02)
    assert lr_cosine(51_000, **kwargs) == pytest.approx(3.3e-4, rel=0.02)
    assert lr_cosine(75_500, **kwargs) == pytest.approx(1.4e-4, rel=0.02)
    assert lr_cosine(100_000, **kwargs) == pytest.approx(6.0e-5)


def test_adam_memory_bytes() -> None:
    assert adam_memory_bytes(7e9) == pytest.approx(112e9)
    assert adam_memory_bytes(7e9, bytes_per_value=2) == pytest.approx(56e9)
