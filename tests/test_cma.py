"""CMA-ES: mechanics, and a real training run that must actually learn."""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.cma import CMAES, train_operator
from atom_neural_rl.channel import ChannelParams
from atom_neural_rl.gym import Gym
from atom_neural_rl.operator import NeuralOperator, OperatorConfig
from atom_neural_rl.reward import episode_reward
from atom_neural_rl.waveforms import WaveformProfile


class CMAMechanics(unittest.TestCase):
    def test_minimizes_a_quadratic(self) -> None:
        target = np.array([1.5, -2.0, 0.5])
        es = CMAES(np.zeros(3), sigma0=1.0, seed=0)
        for _ in range(60):
            xs = es.ask()
            fits = [float(np.sum((x - target) ** 2)) for x in xs]
            es.tell(xs, fits)
        self.assertLess(float(np.sum((es.mean - target) ** 2)), 1e-3)

    def test_population_is_even_and_mirrored(self) -> None:
        es = CMAES(np.zeros(10), sigma0=0.5, seed=1)
        self.assertEqual(es.lam % 2, 0)
        xs = es.ask()
        half = es.lam // 2
        # Mirrored: x_i - mean == -(x_{i+half} - mean).
        np.testing.assert_allclose(xs[:half] - es.mean, -(xs[half:] - es.mean), atol=1e-12)


class _FixedChannelGym(Gym):
    """A narrow gym: one profile, one fixed multipath channel, one rate.

    This isolates 'can the stack learn to equalize' from domain randomization,
    so the learning signal is unambiguous and the test is fast and deterministic.
    """

    def __init__(self) -> None:
        super().__init__(catalog=[WaveformProfile("qpsk", sps=4, rolloff=0.35)])
        # channel_seed fixes the multipath realization: one channel to invert,
        # symbols and noise still vary per episode. This is the fine-tune regime.
        self._chan = ChannelParams(snr_db=25.0, multipath_taps=2, multipath_spread=0.45,
                                    cfo_cycles_per_block=0.0, channel_seed=1234)

    def sample_spec(self, rng, n_samples=4096, noise_prob=0.0):
        spec = super().sample_spec(rng, n_samples=n_samples, noise_prob=noise_prob)
        from atom_neural_rl.gym import EpisodeSpec
        from atom_neural_rl.zplane import F0_HZ
        return EpisodeSpec(
            profile=self.catalog[0], channel=self._chan,
            fs_hz=F0_HZ, n_samples=n_samples, seed=spec.seed,
            n_channels=1, is_noise=spec.is_noise,
        )


class TheStackLearns(unittest.TestCase):
    def test_cma_improves_reward_over_a_planted_channel(self) -> None:
        gym = _FixedChannelGym()
        # Warm-start (H == 1 in the responsive interior), not identity (which
        # saturates the radius sigmoid and leaves the search on a flat plateau).
        template = NeuralOperator.warm_start(OperatorConfig.diagonal_for_channels(1, sections=8))
        history = train_operator(
            template, gym, episode_reward,
            generations=18, batch=6, n_samples=1024,
            sigma0=0.3, popsize=12, seed=3,
        )
        # Warm-start baseline coherence reward is ~0 (H == 1); training must lift
        # it clearly (fixed-rate fine-tune reaches ~0.14 coherence gain).
        self.assertGreater(history.best_reward, 0.03,
                           msg=f"training did not learn: {history.validation_reward}")
        # The improvement must be retained, not a single-generation spike that
        # collapses. (A strict late>early trend is not asserted: CMA-ES validation
        # reward can peak then settle, and the exact path varies with the numpy
        # RNG version across the Python matrix.)
        late = float(np.mean(history.validation_reward[-3:]))
        self.assertGreater(late, 0.02,
                           msg=f"learning not retained: {history.validation_reward}")


if __name__ == "__main__":
    unittest.main()
