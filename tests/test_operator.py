"""Operator properties: identity passthrough, phase equivariance, modReLU, packing."""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.operator import (
    C0,
    C1,
    C3,
    NAMED_CONFIGS,
    NeuralOperator,
    modrelu,
)
from atom_neural_rl.zplane import F0_HZ, RHO_MAX


def _random_iq(rng: np.random.Generator, channels: int, n: int) -> np.ndarray:
    return (rng.standard_normal((channels, n)) + 1j * rng.standard_normal((channels, n))) / np.sqrt(2)


class ModReLU(unittest.TestCase):
    def test_identity_at_zero_threshold(self) -> None:
        z = np.array([1 + 2j, -3 + 0.5j, 0 + 0j, 4 - 1j])
        np.testing.assert_allclose(modrelu(z, 0.0), z, atol=1e-15)

    def test_is_phase_equivariant(self) -> None:
        rng = np.random.default_rng(0)
        z = _random_iq(rng, 1, 256)[0]
        phi = 0.9
        left = modrelu(np.exp(1j * phi) * z, -0.3)
        right = np.exp(1j * phi) * modrelu(z, -0.3)
        np.testing.assert_allclose(left, right, atol=1e-12)

    def test_is_nonexpansive(self) -> None:
        rng = np.random.default_rng(1)
        z = _random_iq(rng, 1, 4096)[0]
        for b in (-0.1, -0.5, -1.0):
            self.assertLessEqual(np.max(np.abs(modrelu(z, b))), np.max(np.abs(z)) + 1e-12)


class IdentityOperator(unittest.TestCase):
    def test_identity_reproduces_input(self) -> None:
        rng = np.random.default_rng(2)
        for config in (C0, C1):
            op = NeuralOperator.identity(config)
            iq = _random_iq(rng, config.in_channels, 2048)
            out = op.forward(iq, F0_HZ)
            np.testing.assert_allclose(out, iq, atol=1e-9)

    def test_identity_holds_after_transport(self) -> None:
        rng = np.random.default_rng(3)
        op = NeuralOperator.identity(C1)
        iq = _random_iq(rng, 2, 1024)
        out = op.forward(iq, F0_HZ / 3)
        np.testing.assert_allclose(out, iq, atol=1e-9)


class PhaseEquivariance(unittest.TestCase):
    def test_operator_commutes_with_global_phase(self) -> None:
        rng = np.random.default_rng(4)
        op = NeuralOperator.identity(C1)
        # Perturb into a non-trivial operator via a random adapted vector.
        vec = rng.standard_normal(C1.adapted_dim) * 0.4
        op = op.with_adapted_vector(vec)
        iq = _random_iq(rng, 2, 1024)
        phi = 1.3
        left = op.forward(np.exp(1j * phi) * iq, F0_HZ)
        right = np.exp(1j * phi) * op.forward(iq, F0_HZ)
        np.testing.assert_allclose(left, right, atol=1e-9)


class AdaptedVectorRoundTrip(unittest.TestCase):
    def test_pack_unpack_is_stable(self) -> None:
        rng = np.random.default_rng(5)
        op = NeuralOperator.identity(C3)
        vec = rng.standard_normal(C3.adapted_dim) * 0.5
        op2 = op.with_adapted_vector(vec)
        vec2 = op2.adapted_vector()
        op3 = op2.with_adapted_vector(vec2)
        # The operator produced by re-packing is functionally identical.
        iq = _random_iq(rng, 2, 512)
        np.testing.assert_allclose(op2.forward(iq, F0_HZ), op3.forward(iq, F0_HZ), atol=1e-9)

    def test_any_vector_yields_stable_poles_and_negative_thresholds(self) -> None:
        rng = np.random.default_rng(6)
        op = NeuralOperator.identity(C3)
        for _ in range(20):
            vec = rng.standard_normal(C3.adapted_dim) * 5.0  # deliberately wild
            op2 = op.with_adapted_vector(vec)
            self.assertTrue(np.all(op2.thresholds <= 0.0))
            for row in op2.kernels:
                for kernel in row:
                    self.assertLessEqual(kernel.pole_radius, RHO_MAX + 1e-12)

    def test_adapted_dim_matches_named_configs(self) -> None:
        # Report and sanity-bound the published budgets.
        dims = {name: cfg.adapted_dim for name, cfg in NAMED_CONFIGS.items()}
        self.assertEqual(dims["C0"], 67)   # 1 kernel * (4*16+2+1)
        self.assertEqual(dims["C1"], 134)  # 2 kernels
        self.assertLess(dims["C3"], 2000)


if __name__ == "__main__":
    unittest.main()
