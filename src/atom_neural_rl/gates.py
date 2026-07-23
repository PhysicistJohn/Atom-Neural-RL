"""Promotion gates G1-G4. A weight bank promotes only if all pass.

- **G1 strict improvement.** Paired-differential reward on train-split episodes:
  mean > 0 with a bootstrap 95% CI excluding 0, *and* an effect-size floor on
  the median. Statistical significance without the floor does not promote.
- **G2 non-inferiority.** On a left-out modulation (leave-one-modulation-out),
  the reward's one-sided lower CI must sit above ``-delta``: the bank may not
  quietly regress a modulation it never trained on.
- **G3 honesty probes.** On signal-free episodes the *blind* (hardware) reward of
  an honest operator scores ~0; a single probe exceeding tolerance fails. Honesty
  is a property of the blind path -- the only path that runs where truth is
  absent -- not of the coherence reward, which simply excludes noise.
- **G3b proxy validity.** The blind (hardware) reward must track the coherence
  (truth) reward in sim above a correlation floor, so hardware fine-tuning against
  the proxy is trustworthy. This is the certificate that replaces per-term
  anti-hacking patches.
- **G4 quantized realizability.** The fixed-point table deviates from the float
  response by less than tolerance across FFT sizes, and every pole radius is
  within the fabric cap on the *quantized* kernel.

G5 (twin acceptance, pinning banks by digest) lives in Atom-NeptuneSDR-Twin and
is out of scope for this repository, which owns G1-G4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .fixedpoint import quantize_table
from .gym import Gym
from .reward import blind_episode_reward, proxy_validity
from .zplane import RHO_MAX


@dataclass(frozen=True)
class GateReport:
    name: str
    passed: bool
    detail: str
    evidence: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PromotionReport:
    gates: List[GateReport]

    @property
    def promoted(self) -> bool:
        return all(g.passed for g in self.gates)

    def summary(self) -> str:
        lines = [f"PROMOTED={self.promoted}"]
        for g in self.gates:
            lines.append(f"  [{'PASS' if g.passed else 'FAIL'}] {g.name}: {g.detail}")
        return "\n".join(lines)


def bootstrap_ci(
    samples: np.ndarray, confidence: float = 0.95, resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    samples = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = samples.size
    if n == 0:
        return (0.0, 0.0)
    means = np.array([rng.choice(samples, n, replace=True).mean() for _ in range(resamples)])
    alpha = (1 - confidence) / 2
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def _rewards(operator, gym: Gym, reward_fn: Callable, count: int, seed: int,
             n_samples: int, noise_prob: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        ep = gym.realize(gym.sample_spec(rng, n_samples=n_samples, noise_prob=noise_prob))
        out.append(reward_fn(operator, ep))
    return np.asarray(out, dtype=np.float64)


def gate_g1(rewards: np.ndarray, floor: float, seed: int = 1) -> GateReport:
    mean = float(np.mean(rewards))
    median = float(np.median(rewards))
    lo, hi = bootstrap_ci(rewards, seed=seed)
    passed = mean > 0.0 and lo > 0.0 and median >= floor
    return GateReport(
        "G1-strict-improvement",
        passed,
        f"mean={mean:.4f} median={median:.4f} CI=({lo:.4f},{hi:.4f}) floor={floor}",
        {"mean": mean, "median": median, "ci_lo": lo, "ci_hi": hi},
    )


def gate_g2(heldout_rewards: np.ndarray, delta: float, seed: int = 2) -> GateReport:
    lo, hi = bootstrap_ci(heldout_rewards, seed=seed)
    mean = float(np.mean(heldout_rewards))
    passed = lo > -delta
    return GateReport(
        "G2-noninferiority",
        passed,
        f"held-out mean={mean:.4f} CI_lo={lo:.4f} > -delta={-delta}",
        {"mean": mean, "ci_lo": lo},
    )


def gate_g3(probe_rewards: np.ndarray, eps: float) -> GateReport:
    # Honesty is one-sided: the violation is *fabricated improvement* (a positive
    # reward on a signal-free episode). A negative probe reward means the operator
    # is penalized on noise (e.g. an equalizer boosting a nulled band amplifies
    # noise) -- a performance tradeoff, not dishonesty, so it does not fail here.
    worst = float(np.max(probe_rewards)) if probe_rewards.size else 0.0
    passed = worst < eps
    return GateReport(
        "G3-honesty-probes",
        passed,
        f"max probe reward={worst:.5f} < eps={eps} over {probe_rewards.size} probes",
        {"worst": worst},
    )


def gate_proxy(correlation: float, floor: float = 0.35) -> GateReport:
    passed = correlation >= floor
    return GateReport(
        "G3b-proxy-validity",
        passed,
        f"corr(truth, blind)={correlation:.3f} >= floor={floor}",
        {"correlation": correlation},
    )


def gate_g4(operator, sizes=(1024, 4096, 16384), tol_db: float = -60.0,
            pole_cap: float = RHO_MAX) -> GateReport:
    worst_dev_db = -np.inf
    worst_radius = 0.0
    for row in operator.kernels:
        for kernel in row:
            worst_radius = max(worst_radius, kernel.pole_radius)
            for n in sizes:
                table = kernel.response_on_grid(n)
                q = quantize_table(table).reconstruct()
                sig = float(np.mean(np.abs(table) ** 2))
                err = float(np.mean(np.abs(table - q) ** 2))
                dev_db = 10 * np.log10(err / sig) if sig > 0 and err > 0 else -np.inf
                worst_dev_db = max(worst_dev_db, dev_db)
    passed = worst_dev_db < tol_db and worst_radius <= pole_cap + 1e-9
    return GateReport(
        "G4-quantized-realizability",
        passed,
        f"worst table dev={worst_dev_db:.1f} dB < {tol_db} dB; worst pole r={worst_radius:.4f} <= {pole_cap:.4f}",
        {"worst_dev_db": worst_dev_db, "worst_radius": worst_radius},
    )


def run_gates(
    operator,
    gym: Gym,
    held_out_modulation: str,
    reward_fn: Callable,
    eval_count: int = 40,
    probe_count: int = 24,
    n_samples: int = 2048,
    g1_floor: float = 0.01,
    g2_delta: float = 0.02,
    g3_eps: float = 0.02,
    seed: int = 0,
) -> PromotionReport:
    """Evaluate all four gates and return the aggregate promotion decision."""
    train_gym, held_gym = gym.leave_one_modulation_out(held_out_modulation)
    # G1/G2 on the coherence (truth) reward.
    train_rewards = _rewards(operator, train_gym, reward_fn, eval_count, seed + 1, n_samples)
    held_rewards = _rewards(operator, held_gym, reward_fn, eval_count, seed + 2, n_samples)
    # G3 honesty on the blind (hardware) reward -- the path that runs without truth.
    probe_rewards = _rewards(
        operator, train_gym, blind_episode_reward, probe_count, seed + 3, n_samples, noise_prob=1.0
    )
    # G3b: the blind proxy must track the truth reward across the operator space.
    correlation = proxy_validity(operator, train_gym, seed=seed + 4, n_samples=n_samples)
    return PromotionReport(
        gates=[
            gate_g1(train_rewards, g1_floor, seed=seed + 11),
            gate_g2(held_rewards, g2_delta, seed=seed + 12),
            gate_g3(probe_rewards, g3_eps),
            gate_proxy(correlation),
            gate_g4(operator),
        ]
    )
