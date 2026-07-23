"""Gym determinism, Nyquist conditioning, and the train/held-out split."""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.gym import FS_MIN_HZ, Gym, default_catalog
from atom_neural_rl.waveforms import MODULATIONS, WaveformProfile, rrc_taps, synthesize
from atom_neural_rl.zplane import F0_HZ


class Waveforms(unittest.TestCase):
    def test_synthesis_is_deterministic(self) -> None:
        p = WaveformProfile("16qam", sps=4, rolloff=0.35)
        a = synthesize(p, 2048, seed=7).iq
        b = synthesize(p, 2048, seed=7).iq
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        p = WaveformProfile("qpsk", sps=4)
        a = synthesize(p, 1024, seed=1).iq
        b = synthesize(p, 1024, seed=2).iq
        self.assertGreater(np.max(np.abs(a - b)), 1e-3)

    def test_rrc_taps_unit_energy(self) -> None:
        taps = rrc_taps(sps=4, rolloff=0.35, span=8)
        self.assertAlmostEqual(float(np.sum(taps ** 2)), 1.0, places=9)

    def test_output_is_unit_power(self) -> None:
        p = WaveformProfile("64qam", sps=8, rolloff=0.5)
        iq = synthesize(p, 4096, seed=3).iq
        self.assertAlmostEqual(float(np.mean(np.abs(iq) ** 2)), 1.0, places=6)


class EpisodeInvariants(unittest.TestCase):
    def test_catalog_covers_all_modulations(self) -> None:
        mods = {p.modulation for p in default_catalog()}
        self.assertEqual(mods, set(MODULATIONS))

    def test_bandwidth_below_nyquist_always(self) -> None:
        gym = Gym()
        rng = np.random.default_rng(0)
        for _ in range(200):
            spec = gym.sample_spec(rng)
            self.assertLess(spec.occupied_bandwidth_hz(), spec.fs_hz)
            self.assertLessEqual(spec.fs_hz, F0_HZ)
            self.assertGreaterEqual(spec.fs_hz, FS_MIN_HZ - 1.0)

    def test_realize_is_deterministic_from_spec(self) -> None:
        gym = Gym()
        rng = np.random.default_rng(1)
        spec = gym.sample_spec(rng)
        a = gym.realize(spec)
        b = gym.realize(spec)
        np.testing.assert_array_equal(a.observed, b.observed)

    def test_channels_are_a_tensor_dimension(self) -> None:
        gym = Gym(n_channels=3)
        rng = np.random.default_rng(2)
        ep = gym.realize(gym.sample_spec(rng, n_samples=1024))
        self.assertEqual(ep.observed.shape, (3, 1024))
        self.assertEqual(ep.clean.shape, (3, 1024))

    def test_noise_episode_has_no_clean_signal(self) -> None:
        gym = Gym()
        rng = np.random.default_rng(3)
        spec = gym.sample_spec(rng, noise_prob=1.0)
        self.assertTrue(spec.is_noise)
        ep = gym.realize(spec)
        self.assertIsNone(ep.synth)
        np.testing.assert_array_equal(ep.clean, 0.0)

    def test_leave_one_modulation_out(self) -> None:
        gym = Gym()
        train, held = gym.leave_one_modulation_out("qpsk")
        self.assertTrue(all(p.modulation != "qpsk" for p in train.catalog))
        self.assertTrue(all(p.modulation == "qpsk" for p in held.catalog))


if __name__ == "__main__":
    unittest.main()
