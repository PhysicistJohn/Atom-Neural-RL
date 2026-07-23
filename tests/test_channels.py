"""Channels are a free tensor dimension: the operator works for any channel count."""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.operator import NeuralOperator, OperatorConfig
from atom_neural_rl.zplane import F0_HZ


def _random_iq(rng: np.random.Generator, channels: int, n: int) -> np.ndarray:
    return (rng.standard_normal((channels, n)) + 1j * rng.standard_normal((channels, n))) / np.sqrt(2)


class ArbitraryChannelCount(unittest.TestCase):
    def test_identity_passthrough_for_many_channels(self) -> None:
        rng = np.random.default_rng(0)
        for channels in (1, 2, 4, 8):
            cfg = OperatorConfig.diagonal_for_channels(channels, sections=8)
            op = NeuralOperator.identity(cfg)
            iq = _random_iq(rng, channels, 512)
            np.testing.assert_allclose(op.forward(iq, F0_HZ), iq, atol=1e-9)

    def test_shape_is_out_channels_by_n(self) -> None:
        rng = np.random.default_rng(1)
        cfg = OperatorConfig.diagonal_for_channels(4, sections=8)
        op = NeuralOperator.identity(cfg)
        vec = rng.standard_normal(cfg.adapted_dim) * 0.3
        op = op.with_adapted_vector(vec)
        out = op.forward(_random_iq(rng, 4, 256), F0_HZ)
        self.assertEqual(out.shape, (4, 256))

    def test_phase_equivariance_holds_per_channel(self) -> None:
        rng = np.random.default_rng(2)
        cfg = OperatorConfig.diagonal_for_channels(3, sections=8)
        op = NeuralOperator.identity(cfg).with_adapted_vector(
            rng.standard_normal(cfg.adapted_dim) * 0.4
        )
        iq = _random_iq(rng, 3, 512)
        phi = 0.77
        left = op.forward(np.exp(1j * phi) * iq, F0_HZ)
        right = np.exp(1j * phi) * op.forward(iq, F0_HZ)
        np.testing.assert_allclose(left, right, atol=1e-9)

    def test_wide_channel_count_adapted_dim_scales_linearly(self) -> None:
        d2 = OperatorConfig.diagonal_for_channels(2, sections=8).adapted_dim
        d8 = OperatorConfig.diagonal_for_channels(8, sections=8).adapted_dim
        self.assertEqual(d8, 4 * d2)


if __name__ == "__main__":
    unittest.main()
