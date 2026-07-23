"""Fixed-point datapath model: quantization bounds, determinism, exponent handling."""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.fixedpoint import (
    WEIGHT_FRAC_BITS,
    fixed_point_spectral_multiply,
    modrelu_fixed,
    quantize_table,
    round_block_float,
    weight_quant_error_dbc,
)
from atom_neural_rl.operator import modrelu
from atom_neural_rl.zplane import make_notch, make_resonator


class TableQuantization(unittest.TestCase):
    def test_reconstruct_within_one_lsb(self) -> None:
        k = make_resonator(radius=0.9, angle=0.6)
        table = k.response_on_grid(4096)
        q = quantize_table(table)
        lsb = 2.0 ** (q.exponent - WEIGHT_FRAC_BITS)
        err = np.max(np.abs(table - q.reconstruct()))
        self.assertLessEqual(err, lsb * np.sqrt(2) + 1e-15)

    def test_gain_above_unity_uses_positive_exponent(self) -> None:
        k = make_resonator(radius=0.95, angle=0.4, gain=6.0 + 0j)
        table = k.response_on_grid(1024)
        q = quantize_table(table)
        self.assertGreaterEqual(q.exponent, 3)  # ~ ceil(log2(peak)) for a sharp resonator
        # Still reconstructs the large-gain peak faithfully.
        self.assertLess(np.max(np.abs(table - q.reconstruct())) / np.max(np.abs(table)), 1e-3)

    def test_weight_quant_error_is_low(self) -> None:
        k = make_notch(radius=0.9, angle=1.0, depth=0.9)
        table = k.response_on_grid(4096)
        dbc = weight_quant_error_dbc(table)
        self.assertLess(dbc, -80.0)  # design target ~ -93 dBc

    def test_quantization_is_deterministic(self) -> None:
        table = make_resonator(radius=0.88, angle=0.5).response_on_grid(2048)
        a = quantize_table(table).reconstruct()
        b = quantize_table(table).reconstruct()
        np.testing.assert_array_equal(a, b)


class BlockFloat(unittest.TestCase):
    def test_round_block_float_is_bounded(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000) + 1j * rng.standard_normal(1000)
        y = round_block_float(x, data_bits=24)
        peak = np.max(np.abs(x))
        lsb = 2.0 ** (int(np.ceil(np.log2(peak))) - 23)
        self.assertLess(np.max(np.abs(x - y)), lsb)

    def test_spectral_multiply_matches_float_within_quant(self) -> None:
        k = make_resonator(radius=0.9, angle=0.7)
        table = k.response_on_grid(2048)
        rng = np.random.default_rng(1)
        spectrum = rng.standard_normal(2048) + 1j * rng.standard_normal(2048)
        fixed = fixed_point_spectral_multiply(quantize_table(table), spectrum)
        exact = table * spectrum
        rel = np.max(np.abs(fixed - exact)) / np.max(np.abs(exact))
        self.assertLess(rel, 1e-3)


class FixedModReLU(unittest.TestCase):
    def test_matches_float_modrelu_closely(self) -> None:
        rng = np.random.default_rng(2)
        z = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
        f = modrelu(z, -0.3)
        q = modrelu_fixed(z, -0.3)
        self.assertLess(np.max(np.abs(f - q)), 1e-5)

    def test_decision_independent_of_block_scale(self) -> None:
        # Scaling the whole block and the threshold together must not change which
        # samples survive -- the exponent-compensation property.
        rng = np.random.default_rng(3)
        z = rng.standard_normal(500) + 1j * rng.standard_normal(500)
        b = -0.5
        survived_1 = np.abs(modrelu_fixed(z, b)) > 0
        survived_2 = np.abs(modrelu_fixed(4.0 * z, 4.0 * b)) > 0
        np.testing.assert_array_equal(survived_1, survived_2)


if __name__ == "__main__":
    unittest.main()
