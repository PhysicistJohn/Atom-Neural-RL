"""Unit coverage for the rational-kernel mechanics themselves."""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.zplane import (
    RHO_MAX,
    RationalKernel,
    _poly_from_roots_zinv,
    make_notch,
    make_resonator,
)


class ProductFormMechanics(unittest.TestCase):
    def test_identity_is_unit_everywhere(self) -> None:
        ident = RationalKernel.identity()
        omega = np.linspace(-np.pi, np.pi, 257)
        np.testing.assert_allclose(ident.evaluate(omega), 1.0 + 0j, atol=0)

    def test_poles_outside_disk_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RationalKernel(1.0 + 0j, np.array([1.01 + 0j]), np.zeros(0))

    def test_polynomial_from_roots_is_monic_in_z0(self) -> None:
        coeff = _poly_from_roots_zinv(np.array([0.5 + 0j, -0.3j]))
        self.assertAlmostEqual(coeff[0], 1.0)
        # (1 - 0.5 z^-1)(1 + 0.3j z^-1) = 1 + (0.3j - 0.5) z^-1 - 0.15j z^-2
        self.assertAlmostEqual(coeff[1], (0.3j - 0.5))
        self.assertAlmostEqual(coeff[2], -0.15j)

    def test_impulse_response_matches_ifft_of_dense_grid(self) -> None:
        k = make_resonator(radius=0.8, angle=0.9)
        n = 4096
        h_direct = k.impulse_response(n)
        h_fft = np.fft.ifft(k.response_on_grid(n))
        # On a grid long enough that aliasing is negligible, they agree.
        self.assertLess(np.max(np.abs(h_direct - h_fft)), 1e-9)


class StabilityProjection(unittest.TestCase):
    def test_projection_pulls_poles_to_cap(self) -> None:
        # Construct just inside the disk, then verify projection clamps to RHO_MAX.
        k = RationalKernel(1.0 + 0j, np.array([0.999 * np.exp(1j * 0.4)]), np.zeros(0))
        projected = k.project_stable(RHO_MAX)
        self.assertLessEqual(projected.pole_radius, RHO_MAX + 1e-15)
        self.assertAlmostEqual(projected.pole_radius, RHO_MAX, places=9)
        # Angle preserved by the radial projection.
        self.assertAlmostEqual(
            float(np.angle(projected.poles[0])), 0.4, places=9
        )

    def test_projection_is_noop_inside_cap(self) -> None:
        k = make_resonator(radius=0.5, angle=0.2)
        projected = k.project_stable(RHO_MAX)
        np.testing.assert_allclose(projected.poles, k.poles, atol=0)


class Residues(unittest.TestCase):
    def test_residue_reconstruction_matches_response(self) -> None:
        k = make_resonator(radius=0.85, angle=0.6, gain=1.3 + 0.2j)
        r, poles, c0 = k.residues()
        omega = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        z_inv = np.exp(-1j * omega)
        recon = c0 + sum(rj / (1.0 - pj * z_inv) for rj, pj in zip(r, poles))
        np.testing.assert_allclose(recon, k.evaluate(omega), atol=1e-9)

    def test_residue_norm_is_positive_for_resonator(self) -> None:
        k = make_resonator(radius=0.9, angle=0.5)
        self.assertGreater(k.residue_norm(), 0.0)


if __name__ == "__main__":
    unittest.main()
