"""Capture ingestion + host-side field deployment with the validation gate."""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.capture import Capture, GymCaptureSource, decode_iq16, encode_iq16
from atom_neural_rl.channel import ChannelParams
from atom_neural_rl.field import finetune_known_signal, run_on_capture
from atom_neural_rl.gym import Gym, EpisodeSpec
from atom_neural_rl.operator import NeuralOperator, OperatorConfig
from atom_neural_rl.waveforms import WaveformProfile
from atom_neural_rl.zplane import F0_HZ


class Iq16ByteFormat(unittest.TestCase):
    def test_round_trip_within_quantization(self) -> None:
        rng = np.random.default_rng(0)
        iq = (rng.standard_normal((2, 4096)) + 1j * rng.standard_normal((2, 4096))) / 8
        restored = decode_iq16(encode_iq16(iq), channels=2)
        self.assertEqual(restored.shape, iq.shape)
        self.assertLess(np.max(np.abs(iq - restored)), 1.0 / 2048)  # < 1 LSB

    def test_channel_interleave_is_time_major(self) -> None:
        # Two channels, distinguishable constant values; check the byte order is
        # I0 Q0 I1 Q1 per sample.
        iq = np.array([[0.5 + 0j, 0.5 + 0j], [0.25 + 0j, 0.25 + 0j]])
        data = np.frombuffer(encode_iq16(iq), dtype="<i2")
        # sample 0: I0=1024 Q0=0 I1=512 Q1=0
        self.assertEqual(list(data[:4]), [1024, 0, 512, 0])


class HostSideExecution(unittest.TestCase):
    def test_run_on_capture_reports_quality(self) -> None:
        gym = Gym()
        src = GymCaptureSource(gym, seed=1)
        cap = src.capture(2048)
        op = NeuralOperator.identity(OperatorConfig.diagonal_for_channels(1))
        result = run_on_capture(op, cap)
        self.assertEqual(result.output.shape, cap.iq.shape)
        # Identity: true coherence gain ~0.
        self.assertIsNotNone(result.true_coherence_gain)
        self.assertAlmostEqual(result.true_coherence_gain, 0.0, delta=1e-6)


class _KnownSignalCaptures:
    """A bag of captures from one fixed channel, each carrying its reference."""

    @staticmethod
    def make(n, seed0):
        prof = WaveformProfile("qpsk", sps=4, rolloff=0.35)
        chan = ChannelParams(snr_db=22.0, multipath_taps=3, multipath_spread=0.6,
                             cfo_cycles_per_block=0.0, channel_seed=4242)
        gym = Gym(catalog=[prof])
        caps = []
        for k in range(n):
            rng = np.random.default_rng(seed0 + k)
            base = gym.sample_spec(rng, n_samples=1024)
            spec = EpisodeSpec(profile=prof, channel=chan, fs_hz=F0_HZ, n_samples=1024,
                               seed=base.seed, n_channels=1)
            ep = gym.realize(spec)
            caps.append(Capture(iq=ep.observed, fs_hz=F0_HZ, sps=4, clean=ep.clean))
        return caps


class KnownSignalFineTune(unittest.TestCase):
    def test_finetune_improves_and_is_validated(self) -> None:
        # The reliable day-one path: known-reference captures -> coherence
        # fine-tune -> validation gate on held-out captures.
        train = _KnownSignalCaptures.make(6, seed0=0)
        val = _KnownSignalCaptures.make(4, seed0=1000)
        template = NeuralOperator.warm_start(OperatorConfig.diagonal_for_channels(1, sections=8))
        outcome = finetune_known_signal(template, train, val, generations=14,
                                        batch=6, n_samples=1024, seed=3)
        self.assertTrue(outcome.accepted, msg=f"validation gain {outcome.validation_gain:.4f}")
        self.assertGreater(outcome.validation_gain, 0.01)

    def test_validation_gate_rejects_a_non_improving_finetune(self) -> None:
        # If training is starved (0 generations), the candidate == warm start,
        # validation gain ~0, and the gate must reject rather than ship it.
        train = _KnownSignalCaptures.make(4, seed0=0)
        val = _KnownSignalCaptures.make(4, seed0=1000)
        template = NeuralOperator.warm_start(OperatorConfig.diagonal_for_channels(1, sections=8))
        outcome = finetune_known_signal(template, train, val, generations=0, seed=3)
        self.assertFalse(outcome.accepted)


if __name__ == "__main__":
    unittest.main()
