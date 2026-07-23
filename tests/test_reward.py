"""Reward stack: anchoring, and rejection of the two headline hacking routes."""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.gym import Gym
from atom_neural_rl.recovery import alignment_error
from atom_neural_rl.reward import RewardConfig, episode_reward, stream_reward
from atom_neural_rl.operator import NeuralOperator, OperatorConfig
from atom_neural_rl.zplane import F0_HZ


class AlignmentError(unittest.TestCase):
    def test_perfect_match_is_zero(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(512) + 1j * rng.standard_normal(512)
        self.assertLess(alignment_error(x, x), 1e-12)

    def test_scale_and_phase_invariant(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal(512) + 1j * rng.standard_normal(512)
        scaled = 7.0 * np.exp(1j * 1.1) * x
        self.assertLess(alignment_error(scaled, x), 1e-9)

    def test_zero_estimate_is_worst(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.standard_normal(256) + 1j * rng.standard_normal(256)
        self.assertEqual(alignment_error(np.zeros(256, dtype=complex), x), 1.0)


class RewardHackingRoutes(unittest.TestCase):
    def _stream(self, op, by, clean, obs, sps=4):
        return stream_reward(op, by, clean, obs, sps)

    def test_perfect_operator_beats_bypass(self) -> None:
        # operator_out == clean should score strongly positive vs a distorted bypass.
        rng = np.random.default_rng(3)
        clean = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)) / np.sqrt(2)
        distorted = clean + 0.5 * (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)) / np.sqrt(2)
        reward = self._stream(clean, distorted, clean, distorted)
        self.assertGreater(reward, 0.1)

    def test_gain_inflation_is_not_rewarded(self) -> None:
        # Scaling the input by a big constant must not earn reward (anchor is
        # scale-invariant; the power penalty punishes the inflation).
        rng = np.random.default_rng(4)
        clean = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)) / np.sqrt(2)
        observed = clean + 0.3 * (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)) / np.sqrt(2)
        inflated = 12.0 * observed
        reward = self._stream(inflated, observed, clean, observed)
        self.assertLessEqual(reward, 0.02)

    def test_content_collapse_is_rejected(self) -> None:
        # Replacing the signal with near-zero (or a lone tone) must score negative.
        rng = np.random.default_rng(5)
        clean = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)) / np.sqrt(2)
        observed = clean + 0.3 * (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)) / np.sqrt(2)
        collapsed = 1e-3 * observed  # signal thrown away
        self.assertLess(self._stream(collapsed, observed, clean, observed), 0.0)

    def test_diverged_output_scores_worst(self) -> None:
        clean = np.ones(64, dtype=complex)
        observed = np.ones(64, dtype=complex)
        bad = np.full(64, np.nan + 1j * np.nan)
        self.assertLessEqual(self._stream(bad, observed, clean, observed), -1.0)


class ProbeAndIdentity(unittest.TestCase):
    def test_identity_operator_scores_near_zero_on_signal(self) -> None:
        gym = Gym()
        rng = np.random.default_rng(6)
        ep = gym.realize(gym.sample_spec(rng, n_samples=2048))
        op = NeuralOperator.identity(OperatorConfig.diagonal_for_channels(1))
        # Identity: op_out == observed == bypass, so the differential is ~0.
        self.assertAlmostEqual(episode_reward(op, ep), 0.0, delta=1e-6)

    def test_identity_probe_is_zero_on_noise(self) -> None:
        gym = Gym()
        rng = np.random.default_rng(7)
        spec = gym.sample_spec(rng, n_samples=2048, noise_prob=1.0)
        ep = gym.realize(spec)
        op = NeuralOperator.identity(OperatorConfig.diagonal_for_channels(1))
        self.assertAlmostEqual(episode_reward(op, ep), 0.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
