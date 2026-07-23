"""Promotion gates: bootstrap CI, realizability, and end-to-end promotion logic."""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.gates import (
    bootstrap_ci,
    gate_g1,
    gate_g3,
    gate_g4,
    run_gates,
)
from atom_neural_rl.gym import Gym
from atom_neural_rl.operator import NeuralOperator, OperatorConfig
from atom_neural_rl.reward import episode_reward


class BootstrapCI(unittest.TestCase):
    def test_ci_brackets_the_mean(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.normal(0.5, 0.1, 200)
        lo, hi = bootstrap_ci(samples, seed=1)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertGreater(lo, 0.0)  # clearly positive mean


class IndividualGates(unittest.TestCase):
    def test_g1_rejects_zero_improvement(self) -> None:
        rewards = np.zeros(30)
        self.assertFalse(gate_g1(rewards, floor=0.01).passed)

    def test_g1_accepts_clear_improvement(self) -> None:
        rng = np.random.default_rng(1)
        rewards = rng.normal(0.1, 0.02, 40)
        self.assertTrue(gate_g1(rewards, floor=0.02).passed)

    def test_g3_single_violation_fails(self) -> None:
        probes = np.array([0.001, 0.002, 0.05, 0.001])  # one hot probe
        self.assertFalse(gate_g3(probes, eps=0.02).passed)

    def test_g3_clean_probes_pass(self) -> None:
        probes = np.array([0.001, -0.002, 0.003, 0.0])
        self.assertTrue(gate_g3(probes, eps=0.02).passed)

    def test_g4_identity_operator_is_realizable(self) -> None:
        op = NeuralOperator.identity(OperatorConfig.diagonal_for_channels(2, sections=8))
        report = gate_g4(op)
        self.assertTrue(report.passed)
        self.assertLessEqual(report.evidence["worst_radius"], 0.9921875 + 1e-9)


class EndToEndPromotion(unittest.TestCase):
    def test_identity_operator_does_not_promote(self) -> None:
        # An identity operator offers no improvement, so G1 must fail and the bank
        # must not promote -- the gate suite is not rubber-stamping.
        gym = Gym()
        op = NeuralOperator.identity(OperatorConfig.diagonal_for_channels(1, sections=8))
        report = run_gates(
            op, gym, held_out_modulation="qpsk", reward_fn=episode_reward,
            eval_count=16, probe_count=8, n_samples=1024, seed=0,
        )
        self.assertFalse(report.promoted)
        g1 = next(g for g in report.gates if g.name.startswith("G1"))
        self.assertFalse(g1.passed)
        # But honesty and realizability should still pass for identity.
        g3 = next(g for g in report.gates if g.name.startswith("G3"))
        g4 = next(g for g in report.gates if g.name.startswith("G4"))
        self.assertTrue(g3.passed)
        self.assertTrue(g4.passed)


if __name__ == "__main__":
    unittest.main()
