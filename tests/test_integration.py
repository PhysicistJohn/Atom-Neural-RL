"""End-to-end: train -> compile bank -> reload -> promote through G1/G3/G4.

This is the P1 acceptance test: a bank trained on a channel must, after
compilation and reload, demonstrably improve signal quality on that channel's
distribution, pass the honesty and realizability gates, and carry a stable
digest. It exercises every module together.
"""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.bank import compile_bank, load_bank
from atom_neural_rl.channel import ChannelParams
from atom_neural_rl.cli import main as cli_main
from atom_neural_rl.cma import train_operator
from atom_neural_rl.gates import gate_g1, gate_g3, gate_g4, _rewards
from atom_neural_rl.gym import Gym, EpisodeSpec
from atom_neural_rl.operator import NeuralOperator, OperatorConfig
from atom_neural_rl.reward import episode_reward
from atom_neural_rl.waveforms import WaveformProfile
from atom_neural_rl.zplane import F0_HZ


class _FixedGym(Gym):
    """Single-rate, single-channel fine-tune regime: one channel to invert at one
    rate. This is the reliable, deterministic hardware fine-tune scenario, as
    opposed to the harder rate-agnostic generalization (a separate claim)."""

    def __init__(self) -> None:
        super().__init__(catalog=[WaveformProfile("qpsk", sps=4, rolloff=0.35)])
        self._c = ChannelParams(snr_db=25.0, multipath_taps=3, multipath_spread=0.6,
                                cfo_cycles_per_block=0.0, channel_seed=2024)

    def sample_spec(self, rng, n_samples=4096, noise_prob=0.0):
        s = super().sample_spec(rng, n_samples=n_samples, noise_prob=noise_prob)
        return EpisodeSpec(profile=self.catalog[0], channel=self._c, fs_hz=F0_HZ,
                           n_samples=n_samples, seed=s.seed, n_channels=1, is_noise=s.is_noise)


class EndToEnd(unittest.TestCase):
    def test_train_compile_reload_and_gate(self) -> None:
        gym = _FixedGym()
        template = NeuralOperator.warm_start(OperatorConfig.diagonal_for_channels(1, sections=8))
        history = train_operator(template, gym, episode_reward,
                                 generations=22, batch=6, n_samples=1024,
                                 sigma0=0.3, popsize=12, seed=5)
        operator = template.with_adapted_vector(history.best_vector)

        # Compile, reload, and confirm the reloaded operator behaves identically.
        bank = compile_bank(operator)
        self.assertTrue(bank.verify_crc())
        reloaded = load_bank(bank.payload).operator
        rng = np.random.default_rng(1)
        iq = (rng.standard_normal((1, 512)) + 1j * rng.standard_normal((1, 512))) / np.sqrt(2)
        fs = gym.sample_spec(rng).fs_hz  # one rate, used for both operators
        np.testing.assert_allclose(operator.forward(iq, fs),
                                   reloaded.forward(iq, fs), atol=1e-9)

        # G1: strict improvement on the trained channel distribution.
        train_rewards = _rewards(reloaded, gym, episode_reward, count=40, seed=7, n_samples=1024)
        self.assertTrue(gate_g1(train_rewards, floor=0.01).passed,
                        msg=f"G1 failed: mean={np.mean(train_rewards):.4f}")

        # G3: honesty on signal-free probes.
        probes = _rewards(reloaded, gym, episode_reward, count=16, seed=8,
                          n_samples=1024, noise_prob=1.0)
        self.assertTrue(gate_g3(probes, eps=0.02).passed,
                        msg=f"G3 failed: worst={np.max(np.abs(probes)):.4f}")

        # G4: quantized realizability.
        self.assertTrue(gate_g4(reloaded).passed)


class CliSmoke(unittest.TestCase):
    def test_verify_invariance_exits_zero(self) -> None:
        self.assertEqual(cli_main(["verify-invariance"]), 0)


if __name__ == "__main__":
    unittest.main()
