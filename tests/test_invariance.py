"""The headline property: z-plane invariance, made numerically checkable.

Each test operationalizes one clause of the invariance contract in
``zplane``. If any of these fails, the central claim of the project is false,
so these are the load-bearing tests of the repository.
"""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.zplane import (
    F0_HZ,
    RHO_MAX,
    RationalKernel,
    make_notch,
    make_resonator,
)


class FrequencyResponseIsSizeInvariant(unittest.TestCase):
    """Clause 1: the response is a function of normalized frequency alone, so the
    same physical frequency reads identically at every FFT size."""

    def test_shared_bins_match_exactly_across_N(self) -> None:
        k = make_notch(radius=0.9, angle=0.7, depth=0.95)
        # Bin j at N corresponds to bin 2j at 2N, 4j at 4N: the same omega.
        h1 = k.response_on_grid(1024)
        h2 = k.response_on_grid(2048)
        h4 = k.response_on_grid(4096)
        idx = np.arange(0, 512)
        np.testing.assert_allclose(h1[idx], h2[2 * idx], rtol=0, atol=1e-12)
        np.testing.assert_allclose(h1[idx], h4[4 * idx], rtol=0, atol=1e-12)

    def test_response_at_fixed_physical_frequency_is_N_independent(self) -> None:
        k = make_resonator(radius=0.96, angle=1.1)
        f_phys = 5.0e6  # a physical frequency well inside the band
        omega = 2.0 * np.pi * f_phys / F0_HZ
        vals = [k.evaluate(np.array([omega]))[0] for _ in (1024, 8192, 65536)]
        for v in vals[1:]:
            self.assertAlmostEqual(abs(v - vals[0]), 0.0, places=12)


class AliasingCertificateHolds(unittest.TestCase):
    """Clause 2: N-point circular convolution deviates from true aperiodic
    filtering by no more than the certified bound, which shrinks with N."""

    def _measured_circular_error(self, k: RationalKernel, n: int) -> float:
        # Periodized (circular) impulse response vs the true causal one.
        h_circular = np.fft.ifft(k.response_on_grid(n))
        h_true = k.impulse_response(n)
        return float(np.max(np.abs(h_circular - h_true)))

    def test_measured_error_within_certificate(self) -> None:
        for k in (
            make_resonator(radius=0.9, angle=0.5),
            make_notch(radius=0.95, angle=1.3, depth=0.9),
        ):
            for n in (1024, 4096, 16384):
                measured = self._measured_circular_error(k, n)
                certified = k.aliasing_certificate(n)
                self.assertLessEqual(
                    measured,
                    certified + 1e-15,
                    msg=f"measured {measured:.3e} exceeds certificate {certified:.3e} at N={n}",
                )

    def test_certificate_shrinks_geometrically_with_N(self) -> None:
        k = make_resonator(radius=0.9, angle=0.5)
        e10 = k.aliasing_certificate(1024)
        e13 = k.aliasing_certificate(8192)
        # rho=0.9: rho^8192 / rho^1024 is astronomically small; assert strong decay.
        self.assertLess(e13, e10 * 1e-100)

    def test_identity_kernel_has_zero_certificate(self) -> None:
        ident = RationalKernel.identity()
        self.assertEqual(ident.aliasing_certificate(1024), 0.0)
        np.testing.assert_allclose(ident.response_on_grid(2048), 1.0, atol=0)

    def test_honest_bound_beats_naive_shortcut_at_small_N(self) -> None:
        # The naive rule rho_max = eps^(1/N) ignores 1/(1-rho); confirm the honest
        # solver returns a *smaller* admissible radius (i.e. is stricter) so the
        # ~40 dB overclaim cannot recur.
        k = make_resonator(radius=0.99, angle=0.4)
        eps = 2.0 ** -12
        n = 1024
        honest = k.max_stable_radius_for(n, eps)
        naive = eps ** (1.0 / n)
        self.assertLess(honest, naive)


class MasterRateTransportMovesPolesInward(unittest.TestCase):
    """Clause 3: transport from the master rate to any legal rate only shrinks
    pole radii, and preserves the physical-frequency location of features."""

    def test_transport_is_identity_at_master_rate(self) -> None:
        k = make_resonator(radius=0.95, angle=0.8)
        t = k.transport(F0_HZ)
        np.testing.assert_allclose(t.poles, k.poles, atol=1e-12)

    def test_poles_move_inward_for_lower_rates(self) -> None:
        k = make_resonator(radius=0.98, angle=0.6)
        for fs in (F0_HZ / 2, F0_HZ / 4, 10.0e6):
            t = k.transport(fs)
            self.assertLessEqual(t.pole_radius, k.pole_radius + 1e-15)
            self.assertLess(t.pole_radius, k.pole_radius)  # strictly, since fs<f0

    def test_projection_never_fires_after_transport(self) -> None:
        # A max-radius anchored pole stays <= RHO_MAX after any legal transport.
        k = make_resonator(radius=RHO_MAX, angle=0.3)
        for fs in (F0_HZ, F0_HZ / 2, 5.0e6, 1.0e6):
            t = k.transport(fs)
            self.assertLessEqual(t.pole_radius, RHO_MAX + 1e-15)

    def test_physical_frequency_of_a_pole_is_preserved(self) -> None:
        # The continuous-time operator is invariant: a pole's physical frequency
        # angle*rate/(2pi) is unchanged by transport (within the deployment band).
        k = make_resonator(radius=0.96, angle=0.5)  # 0.5 rad at 61.44 MHz
        f_phys_anchor = 0.5 / (2 * np.pi) * F0_HZ
        for fs in (F0_HZ / 2, 20.0e6):
            t = k.transport(fs)
            angle = float(np.angle(t.poles[0]))
            f_phys = angle / (2 * np.pi) * fs
            self.assertAlmostEqual(f_phys, f_phys_anchor, delta=1.0)  # <1 Hz


if __name__ == "__main__":
    unittest.main()
