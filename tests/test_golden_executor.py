"""GoldenExecutor: fully-integer operator execution on the golden arithmetic."""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.channel import ChannelParams
from atom_neural_rl.cma import train_operator
from atom_neural_rl.gates import _rewards
from atom_neural_rl.golden_executor import GoldenExecutor
from atom_neural_rl.gym import Gym, EpisodeSpec
from atom_neural_rl.operator import NeuralOperator, OperatorConfig
from atom_neural_rl.reward import episode_reward
from atom_neural_rl.waveforms import WaveformProfile
from atom_neural_rl.zplane import F0_HZ


def _iq(rng, ch, n):
    return (rng.standard_normal((ch, n)) + 1j * rng.standard_normal((ch, n))) / 8


class GoldenExecution(unittest.TestCase):
    def test_warm_start_is_near_identity(self) -> None:
        cfg = OperatorConfig.diagonal_for_channels(1, sections=8)
        gx = GoldenExecutor(NeuralOperator.warm_start(cfg))
        rng = np.random.default_rng(0)
        iq = _iq(rng, 1, 1024)
        out = gx.forward(iq, F0_HZ)
        rel = np.max(np.abs(out - iq)) / np.max(np.abs(iq))
        self.assertLess(rel, 3e-3)  # Q1.15 table + per-stage rounding budget

    def test_tracks_float_operator_on_mild_kernels(self) -> None:
        cfg = OperatorConfig.diagonal_for_channels(1, sections=8)
        rng = np.random.default_rng(1)
        base = NeuralOperator.warm_start(cfg)
        op = base.with_adapted_vector(base.adapted_vector() + rng.standard_normal(cfg.adapted_dim) * 0.15)
        iq = _iq(rng, 1, 1024)
        fo = op.forward(iq, F0_HZ)
        go = GoldenExecutor(op).forward(iq, F0_HZ)
        rel = np.max(np.abs(fo - go)) / np.max(np.abs(fo))
        self.assertLess(rel, 5e-3)

    def test_integer_path_is_deterministic(self) -> None:
        cfg = OperatorConfig.diagonal_for_channels(1, sections=8)
        rng = np.random.default_rng(2)
        base = NeuralOperator.warm_start(cfg)
        op = base.with_adapted_vector(base.adapted_vector() + rng.standard_normal(cfg.adapted_dim) * 0.2)
        gx = GoldenExecutor(op)
        re = np.asarray(np.rint(_iq(rng, 1, 512).real * (1 << 23)), dtype=np.int64)
        im = np.zeros_like(re)
        a = gx.forward_int(re, im, F0_HZ)
        b = gx.forward_int(re, im, F0_HZ)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual(a[2], b[2])


class _FixedGym(Gym):
    def __init__(self) -> None:
        super().__init__(catalog=[WaveformProfile("qpsk", sps=4, rolloff=0.35)])
        self._c = ChannelParams(snr_db=22.0, multipath_taps=3, multipath_spread=0.6,
                                cfo_cycles_per_block=0.0, channel_seed=1234)

    def sample_spec(self, rng, n_samples=4096, noise_prob=0.0):
        s = super().sample_spec(rng, n_samples=n_samples, noise_prob=noise_prob)
        return EpisodeSpec(profile=self.catalog[0], channel=self._c, fs_hz=F0_HZ,
                           n_samples=n_samples, seed=s.seed, n_channels=1, is_noise=s.is_noise)


class LoopClosesOnGoldenArithmetic(unittest.TestCase):
    def test_finetune_improves_coherence_fully_integer(self) -> None:
        # The bit-exactness reference is also a working training target: CMA-ES
        # through the golden integer datapath still closes the coherence loop.
        gym = _FixedGym()
        template = GoldenExecutor(
            NeuralOperator.warm_start(OperatorConfig.diagonal_for_channels(1, sections=8))
        )
        history = train_operator(template, gym, episode_reward, generations=14, batch=6,
                                 n_samples=1024, sigma0=0.3, popsize=12, seed=5)
        best = template.with_adapted_vector(history.best_vector)
        gain = float(np.mean(_rewards(best, gym, episode_reward, count=24, seed=99, n_samples=1024)))
        self.assertGreater(gain, 0.02, msg=f"golden-integer loop did not close: {gain:.4f}")


if __name__ == "__main__":
    unittest.main()
