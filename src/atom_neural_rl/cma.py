"""CMA-ES over the small kernel parameter vector, with the paired-differential loop.

CMA-ES is the normative optimizer: gradient-free (no backprop through hardware),
robust to the noisy paired rewards, and efficient at the ``10^2`` real
parameters the rational-kernel parameterization exposes -- which is the whole
reason that parameterization and the RL are one idea. Mirrored sampling and
common random numbers (the same realized episode batch for every candidate in a
generation) are the variance-reduction levers; the reward is already a
within-episode differential, so the two compose.

This is a compact, dependency-free implementation of (mu/mu_w, lambda)-CMA-ES
following Hansen's reference equations. It minimizes; the training loop passes
``-reward``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


class CMAES:
    """Minimizing (mu/mu_w, lambda)-CMA-ES with mirrored sampling."""

    def __init__(
        self,
        x0: np.ndarray,
        sigma0: float,
        popsize: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.n = int(x0.size)
        self.mean = np.asarray(x0, dtype=np.float64).copy()
        self.sigma = float(sigma0)
        self.rng = np.random.default_rng(seed)

        lam = popsize or (4 + int(3 * np.log(self.n)))
        if lam % 2 == 1:
            lam += 1  # even for mirrored pairs
        self.lam = lam
        self.mu = lam // 2

        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        w /= np.sum(w)
        self.weights = w
        self.mueff = 1.0 / np.sum(w ** 2)

        n = self.n
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff),
        )
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.invsqrtC = np.eye(n)
        self.gen = 0

    def ask(self) -> np.ndarray:
        """Return ``lambda`` candidate rows (mirrored pairs)."""
        half = self.lam // 2
        z = self.rng.standard_normal((half, self.n))
        z = np.concatenate([z, -z], axis=0)  # mirrored
        # einsum, not ``@``: numpy 2.0's matmul kernel emits spurious FP-flag
        # RuntimeWarnings on this platform even for finite real arrays.
        scaled = self.D[:, None] * z.T  # (n, lam)
        y = np.einsum("ij,jk->ik", self.B, scaled).T  # (lam, n)
        self._last = y
        return self.mean + self.sigma * y

    def tell(self, solutions: np.ndarray, fitnesses: Sequence[float]) -> None:
        """Update the distribution from evaluated candidates (minimization)."""
        order = np.argsort(fitnesses)
        y = self._last[order]
        y_mu = y[: self.mu]
        y_w = self.weights @ y_mu  # recombined step (in y-space)
        self.mean = self.mean + self.sigma * y_w

        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * np.einsum("ij,j->i", self.invsqrtC, y_w)
        ps_norm = np.linalg.norm(self.ps)
        hsig = ps_norm / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN < (
            1.4 + 2 / (self.n + 1)
        )
        self.pc = (1 - self.cc) * self.pc + (
            hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff)
        ) * y_w

        artmp = y_mu
        rank_mu = np.einsum("jn,j,jm->nm", artmp, self.weights, artmp)
        delta_hsig = (1 - hsig) * self.cc * (2 - self.cc)
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc) + delta_hsig * self.C)
            + self.cmu * rank_mu
        )

        self.sigma *= np.exp((self.cs / self.damps) * (ps_norm / self.chiN - 1))

        # Eigen-update (n is small; do it every generation for correctness).
        self.C = np.triu(self.C) + np.triu(self.C, 1).T  # enforce symmetry
        eigvals, self.B = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-14)
        self.D = np.sqrt(eigvals)
        self.invsqrtC = np.einsum("ij,j,kj->ik", self.B, 1.0 / self.D, self.B)
        self.gen += 1


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
@dataclass
class TrainHistory:
    """Per-generation validation reward and the best vector found."""

    validation_reward: List[float] = field(default_factory=list)
    best_vector: Optional[np.ndarray] = None
    best_reward: float = -np.inf


def train_operator(
    template,
    gym,
    reward_fn: Callable,
    generations: int = 20,
    batch: int = 8,
    n_samples: int = 1024,
    sigma0: float = 0.3,
    popsize: Optional[int] = None,
    seed: int = 0,
    noise_prob: float = 0.0,
) -> TrainHistory:
    """Train ``template``'s adapted vector by CMA-ES with CRN paired rewards.

    ``template`` supplies the config and frozen backbone; ``reward_fn(operator,
    episode)`` returns the paired-differential reward. A fixed validation batch
    (separate seed) tracks progress without leaking into the search.
    """
    x0 = template.adapted_vector()
    es = CMAES(x0, sigma0, popsize=popsize, seed=seed)
    history = TrainHistory()

    # Fixed validation batch, disjoint from the training seed stream.
    val_rng = np.random.default_rng(seed + 10_000)
    val_specs = [gym.sample_spec(val_rng, n_samples=n_samples) for _ in range(batch)]
    val_eps = [gym.realize(s) for s in val_specs]

    gen_rng = np.random.default_rng(seed + 1)
    for _ in range(generations):
        # Common random numbers: one realized batch for the whole generation.
        specs = [gym.sample_spec(gen_rng, n_samples=n_samples, noise_prob=noise_prob) for _ in range(batch)]
        episodes = [gym.realize(s) for s in specs]

        candidates = es.ask()
        fitnesses = []
        for x in candidates:
            op = template.with_adapted_vector(x)
            reward = float(np.mean([reward_fn(op, ep) for ep in episodes]))
            fitnesses.append(-reward)  # minimize -reward
        es.tell(candidates, fitnesses)

        val_op = template.with_adapted_vector(es.mean)
        val_reward = float(np.mean([reward_fn(val_op, ep) for ep in val_eps]))
        history.validation_reward.append(val_reward)
        if val_reward > history.best_reward:
            history.best_reward = val_reward
            history.best_vector = es.mean.copy()

    if history.best_vector is None:
        history.best_vector = es.mean.copy()
    return history
