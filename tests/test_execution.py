"""ABI-faithful fixed-point execution: fidelity, the CRC gate, and loop closure."""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.bank import compile_bank
from atom_neural_rl.channel import ChannelParams
from atom_neural_rl.cma import train_operator
from atom_neural_rl.execution import AbiExecutor
from atom_neural_rl.gates import _rewards
from atom_neural_rl.gym import Gym, EpisodeSpec
from atom_neural_rl.operator import NeuralOperator, OperatorConfig
from atom_neural_rl.reward import episode_reward
from atom_neural_rl.waveforms import WaveformProfile
from atom_neural_rl.zplane import F0_HZ


def _rand_iq(rng, channels, n):
    return (rng.standard_normal((channels, n)) + 1j * rng.standard_normal((channels, n))) / np.sqrt(2)


class FixedPointFidelity(unittest.TestCase):
    def test_output_tracks_float_within_quantization(self) -> None:
        # Over several mild operators, the fixed-point output stays close to the
        # float reference (relative error set by the Q1.15 table + 24-bit data).
        rng = np.random.default_rng(0)
        cfg = OperatorConfig.diagonal_for_channels(1, sections=8)
        worst = 0.0
        for _ in range(6):
            op = NeuralOperator.warm_start(cfg).with_adapted_vector(
                rng.standard_normal(cfg.adapted_dim) * 0.15
            )
            ex = AbiExecutor.from_operator(op)
            iq = _rand_iq(rng, 1, 2048)
            fo = op.forward(iq, F0_HZ)
            xo = ex.forward(iq, F0_HZ)
            denom = np.max(np.abs(fo))
            if denom < 1e-9:
                continue
            worst = max(worst, float(np.max(np.abs(fo - xo)) / denom))
        self.assertLess(worst, 5e-3, msg=f"fixed-point diverges from float: {worst:.2e}")

    def test_output_is_always_finite(self) -> None:
        rng = np.random.default_rng(1)
        cfg = OperatorConfig.diagonal_for_channels(2, sections=8)
        for _ in range(8):
            op = NeuralOperator.warm_start(cfg).with_adapted_vector(
                rng.standard_normal(cfg.adapted_dim) * 0.5
            )
            xo = AbiExecutor.from_operator(op).forward(_rand_iq(rng, 2, 1024), F0_HZ)
            self.assertTrue(np.all(np.isfinite(xo.view(np.float64))))


class BankContract(unittest.TestCase):
    def test_from_bank_verifies_crc(self) -> None:
        cfg = OperatorConfig.diagonal_for_channels(1, sections=8)
        op = NeuralOperator.warm_start(cfg)
        bank = compile_bank(op)
        ex = AbiExecutor.from_bank(bank)  # clean bank loads
        self.assertIsNotNone(ex.result_weight_crc)
        corrupt = bytearray(bank.payload)
        corrupt[30] ^= 0xFF
        from atom_neural_rl.bank import WeightBank
        bad = WeightBank(manifest=bank.manifest, payload=bytes(corrupt), operator=op)
        with self.assertRaises(ValueError):
            AbiExecutor.from_bank(bad)

    def test_rejects_uncertified_size(self) -> None:
        cfg = OperatorConfig.diagonal_for_channels(1, sections=8)
        op = NeuralOperator.warm_start(cfg).with_adapted_vector(
            np.random.default_rng(3).standard_normal(cfg.adapted_dim) * 0.4
        )
        bank = compile_bank(op)
        ex = AbiExecutor.from_bank(bank)
        if bank.manifest.log2n_min > 10:
            with self.assertRaises(ValueError):
                ex.forward(np.zeros((1, 1 << (bank.manifest.log2n_min - 1)), dtype=complex), F0_HZ)
        else:
            self.skipTest("bank certified to 2^10; no smaller size to reject")


class _FixedGym(Gym):
    def __init__(self) -> None:
        super().__init__(catalog=[WaveformProfile("qpsk", sps=4, rolloff=0.35)])
        self._c = ChannelParams(snr_db=22.0, multipath_taps=3, multipath_spread=0.6,
                                cfo_cycles_per_block=0.0, channel_seed=1234)

    def sample_spec(self, rng, n_samples=4096, noise_prob=0.0):
        s = super().sample_spec(rng, n_samples=n_samples, noise_prob=noise_prob)
        return EpisodeSpec(profile=self.catalog[0], channel=self._c, fs_hz=F0_HZ,
                           n_samples=n_samples, seed=s.seed, n_channels=1, is_noise=s.is_noise)


class LoopClosesThroughTheAbi(unittest.TestCase):
    def test_finetune_improves_coherence_in_fixed_point(self) -> None:
        # CMA-ES trains *through* the fixed-point ABI (reward computed on the
        # quantized output). If coherence still improves, sim-to-hardware transfer
        # is credible: the loop closes through the same contract the twin runs.
        gym = _FixedGym()
        template = AbiExecutor.from_operator(
            NeuralOperator.warm_start(OperatorConfig.diagonal_for_channels(1, sections=8))
        )
        history = train_operator(template, gym, episode_reward, generations=16, batch=6,
                                 n_samples=1024, sigma0=0.3, popsize=12, seed=5)
        best = template.with_adapted_vector(history.best_vector)
        gain = float(np.mean(_rewards(best, gym, episode_reward, count=24, seed=99, n_samples=1024)))
        self.assertGreater(gain, 0.02, msg=f"fixed-point loop did not close: {gain:.4f}")


if __name__ == "__main__":
    unittest.main()
