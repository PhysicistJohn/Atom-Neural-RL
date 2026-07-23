"""The core-truth reward: coherence, and why every hack fails by construction.

These tests assert that the *definition* of the reward -- improvement in
coherence to the clean waveform -- makes the reward-hacking routes impossible,
without any penalty, clip, deadzone, or lock-gate defending it.
"""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.gym import Gym
from atom_neural_rl.operator import NeuralOperator, OperatorConfig
from atom_neural_rl.recovery import coherence
from atom_neural_rl.reward import episode_reward
from atom_neural_rl.zplane import F0_HZ


class CoherenceIsTheCoreTruth(unittest.TestCase):
    def test_perfect_copy_is_one(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
        self.assertAlmostEqual(coherence(x, x), 1.0, places=6)

    def test_invariant_to_gain_phase_delay_cfo(self) -> None:
        rng = np.random.default_rng(1)
        n = 1024
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        gain_phase = 9.0 * np.exp(1j * 2.1) * x
        cfo = x * np.exp(2j * np.pi * 1.5 * np.arange(n) / n)
        self.assertGreater(coherence(gain_phase, x), 0.999)
        self.assertGreater(coherence(np.roll(x, 6), x), 0.99)
        self.assertGreater(coherence(cfo, x), 0.999)

    def test_noise_lowers_coherence_monotonically(self) -> None:
        rng = np.random.default_rng(2)
        n = 2048
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        prev = 1.0
        for snr in (30.0, 20.0, 10.0, 3.0):
            noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
            noise *= np.sqrt(10 ** (-snr / 10))
            g = coherence(x + noise, x)
            self.assertLess(g, prev)
            prev = g

    def test_collapse_and_orthogonal_are_zero(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
        self.assertEqual(coherence(np.zeros(1024, dtype=complex), x), 0.0)
        orthogonal = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
        self.assertLess(coherence(orthogonal, x), 0.05)


class _ScaleOperator:
    """A fake operator that just multiplies its input by a constant (gain hack)."""

    def __init__(self, config, factor):
        self.config = config
        self.factor = factor

    def forward(self, iq, fs_hz):
        return self.factor * np.asarray(iq, dtype=np.complex128)


class _CollapseOperator:
    """Discards the signal structure -- replaces it with a lone constant tone.

    (Merely scaling the input is *not* collapse: coherence is gain-invariant, so
    a uniform scale scores exactly 0. Collapse means destroying the waveform.)
    """

    def __init__(self, config):
        self.config = config

    def forward(self, iq, fs_hz):
        iq = np.asarray(iq, dtype=np.complex128)
        n = iq.shape[-1]
        tone = np.exp(2j * np.pi * 0.1 * np.arange(n))
        return np.broadcast_to(tone, iq.shape).copy()


class HacksFailByConstruction(unittest.TestCase):
    def setUp(self) -> None:
        self.gym = Gym()
        self.rng = np.random.default_rng(4)
        self.ep = self.gym.realize(self.gym.sample_spec(self.rng, n_samples=2048))
        self.cfg = OperatorConfig.diagonal_for_channels(1)

    def test_identity_scores_zero(self) -> None:
        op = NeuralOperator.identity(self.cfg)
        self.assertAlmostEqual(episode_reward(op, self.ep), 0.0, delta=1e-6)

    def test_gain_inflation_earns_nothing(self) -> None:
        # A 20x gain has identical coherence to the input, so reward == 0.
        op = _ScaleOperator(self.cfg, 20.0)
        self.assertAlmostEqual(episode_reward(op, self.ep), 0.0, delta=1e-6)

    def test_content_collapse_is_punished(self) -> None:
        op = _CollapseOperator(self.cfg)
        self.assertLess(episode_reward(op, self.ep), 0.0)

    def test_uniform_scale_is_neither_rewarded_nor_punished(self) -> None:
        # The elegance check: a pure gain is invisible to coherence, so it earns
        # exactly nothing -- no penalty needed to make this true.
        self.assertAlmostEqual(episode_reward(_ScaleOperator(self.cfg, 0.01), self.ep),
                               0.0, delta=1e-6)

    def test_perfect_operator_earns_positive(self) -> None:
        # An operator that outputs the clean waveform maximizes coherence.
        clean = self.ep.clean

        class _Oracle:
            config = self.cfg
            def forward(self, iq, fs_hz):
                return clean
        self.assertGreater(episode_reward(_Oracle(), self.ep), 0.05)

    def test_noise_episode_scores_zero(self) -> None:
        spec = self.gym.sample_spec(self.rng, n_samples=2048, noise_prob=1.0)
        ep = self.gym.realize(spec)
        op = NeuralOperator.identity(self.cfg).with_adapted_vector(
            self.rng.standard_normal(self.cfg.adapted_dim) * 0.3
        )
        # No clean waveform to be faithful to -> reward is exactly 0, for any op.
        self.assertEqual(episode_reward(op, ep), 0.0)

    def test_diverged_output_scores_worst(self) -> None:
        class _Diverged:
            config = self.cfg
            def forward(self, iq, fs_hz):
                return np.full_like(np.asarray(iq, dtype=complex), np.nan)
        self.assertLessEqual(episode_reward(_Diverged(), self.ep), -1.0)


class ProxyMechanismIsSound(unittest.TestCase):
    """The blind proxy tracks the coherence truth along a controlled quality
    ladder -- proven deterministically, without trained-operator sampling noise."""

    def test_both_metrics_rank_a_noise_ladder_consistently(self) -> None:
        from atom_neural_rl.recovery import blind_recover, coherence
        from atom_neural_rl.reward import blind_quality
        from atom_neural_rl.waveforms import WaveformProfile, synthesize

        prof = WaveformProfile("qpsk", sps=4, rolloff=0.35)
        clean = synthesize(prof, 2048, seed=0).iq
        rng = np.random.default_rng(1)
        truth, blind = [], []
        for snr_db in (30, 24, 18, 12, 8, 4):  # decreasing quality
            noise = (rng.standard_normal(clean.size) + 1j * rng.standard_normal(clean.size))
            noise *= np.sqrt(10 ** (-snr_db / 10) / 2)
            z = clean + noise
            truth.append(coherence(z, clean))
            blind.append(blind_quality(z, prof.sps)[0])
        # Both must be (weakly) monotone decreasing with SNR, so their rank
        # correlation is strongly positive.
        tr = np.argsort(np.argsort(truth))
        br = np.argsort(np.argsort(blind))
        r = float(np.corrcoef(tr, br)[0, 1])
        self.assertGreater(r, 0.8, msg=f"blind proxy does not track truth: r={r:.2f}")


if __name__ == "__main__":
    unittest.main()
