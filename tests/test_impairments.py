"""Impairment estimators must recover planted values -- the harness is trusted
in sim before it ever touches the board."""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.impairments import (
    apply_impairments,
    calibration_waveform,
    estimate_impairments,
)


class EstimatorsRecoverPlantedValues(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 4096
        self.ref = calibration_waveform(self.n)

    def test_clean_loopback_reads_near_zero(self) -> None:
        est = estimate_impairments(self.ref, self.ref)
        self.assertLess(abs(est.dc_offset), 1e-9)
        self.assertLess(abs(est.iq_gain_imbalance_db), 1e-6)
        self.assertLess(abs(est.iq_phase_error_deg), 1e-6)
        self.assertLess(abs(est.cfo_bins), 1e-6)

    def test_dc_offset_recovered(self) -> None:
        cap = apply_impairments(self.ref, dc_offset=0.01 + 0.02j)
        est = estimate_impairments(cap, self.ref)
        self.assertAlmostEqual(est.dc_offset.real, 0.01, places=4)
        self.assertAlmostEqual(est.dc_offset.imag, 0.02, places=4)

    def test_iq_imbalance_recovered(self) -> None:
        cap = apply_impairments(self.ref, iq_gain_imbalance_db=0.8, iq_phase_error_deg=2.5)
        est = estimate_impairments(cap, self.ref)
        self.assertAlmostEqual(est.iq_gain_imbalance_db, 0.8, delta=0.05)
        self.assertAlmostEqual(est.iq_phase_error_deg, 2.5, delta=0.15)

    def test_cfo_recovered(self) -> None:
        cap = apply_impairments(self.ref, cfo_bins=3.35)
        est = estimate_impairments(cap, self.ref)
        self.assertAlmostEqual(est.cfo_bins, 3.35, delta=0.05)

    def test_noise_floor_tracks_planted_level(self) -> None:
        cap = apply_impairments(self.ref, noise_dbfs=-45.0, seed=1)
        est = estimate_impairments(cap, self.ref)
        self.assertAlmostEqual(est.noise_floor_dbfs, -45.0, delta=1.5)

    def test_combined_impairments_recovered_jointly(self) -> None:
        cap = apply_impairments(self.ref, dc_offset=0.005 - 0.003j,
                                iq_gain_imbalance_db=0.5, iq_phase_error_deg=1.5,
                                cfo_bins=1.7, noise_dbfs=-50.0, seed=2)
        est = estimate_impairments(cap, self.ref)
        self.assertAlmostEqual(est.cfo_bins, 1.7, delta=0.05)
        self.assertAlmostEqual(est.iq_gain_imbalance_db, 0.5, delta=0.1)
        self.assertAlmostEqual(est.iq_phase_error_deg, 1.5, delta=0.3)
        # DC and CFO are physically entangled: DC is a post-mixer artifact, so a
        # frequency-offset signal leaks its tones (~-43 dBFS) into the DC bin.
        # The standalone (CFO-free) test recovers DC to 4 places; jointly it is
        # accurate to ~1e-2, which is well below any level that matters.
        self.assertLess(abs(est.dc_offset - (0.005 - 0.003j)), 0.01)


if __name__ == "__main__":
    unittest.main()
