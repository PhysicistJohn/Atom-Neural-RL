"""Golden integer arithmetic: correctness, determinism, and pinned digests.

The digests pinned here are the cross-implementation contract: the C core in
the twin and the RTL testbench must reproduce these exact values. Pure integer
arithmetic on committed ROM data -- no libm, no platform variance.
"""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl import golden


class RomIntegrity(unittest.TestCase):
    ROM_SHA256 = "12b5bcfe886e72c759c3a231a46283fc2c4fddadb9fdad92325e1eb8026100f7"

    def test_committed_rom_digest(self) -> None:
        self.assertEqual(golden.rom_sha256(), self.ROM_SHA256)

    def test_rom_matches_regeneration(self) -> None:
        # The generator (float, libm) must agree with the committed artifact on
        # THIS host; the committed file remains the truth if a host ever differs.
        np.testing.assert_array_equal(golden.generate_twiddle_rom(), golden.load_twiddle_rom())


class RoundingRule(unittest.TestCase):
    def test_rhe_ties_to_even(self) -> None:
        # 1/2 -> 0 (even), 3/2 -> 2, 5/2 -> 2, -1/2 -> 0, -3/2 -> -2
        vals = np.array([1, 3, 5, -1, -3, 2, 4, -2])
        expect = np.array([0, 2, 2, 0, -2, 1, 2, -1])
        np.testing.assert_array_equal(golden.rhe(vals, 1), expect)

    def test_rhe_matches_python_round_half_even(self) -> None:
        rng = np.random.default_rng(0)
        v = rng.integers(-1 << 40, 1 << 40, size=2000)
        for s in (1, 5, 15, 17):
            expect = np.array([round(x / (1 << s)) for x in v.tolist()], dtype=np.int64)
            np.testing.assert_array_equal(golden.rhe(v, s), expect)


class FftCorrectness(unittest.TestCase):
    def _float_ref(self, re, im, n):
        x = (re + 1j * im).astype(np.complex128)
        return np.fft.fft(x) / n

    def test_tracks_mathematical_fft(self) -> None:
        for n in (256, 1024, 4096):
            re, im = golden.vector_input(seed=1, n=n)
            gr, gi = golden.golden_fft(re, im)
            ref = self._float_ref(re, im, n)
            err = np.max(np.abs((gr + 1j * gi) - ref))
            # Error budget: per-stage rhe rounding, ~O(sqrt(log2 N)) LSBs of the
            # stage-scaled data; assert well under the 12-bit signal scale.
            self.assertLess(err, 40.0, msg=f"N={n}: max abs err {err:.1f}")
            # And relative to full scale (input <<8, so bins ~2^15..2^20):
            rel = err / np.max(np.abs(ref))
            self.assertLess(rel, 2e-3, msg=f"N={n}: rel err {rel:.2e}")

    def test_impulse_gives_flat_spectrum(self) -> None:
        n = 256
        re = np.zeros(n, dtype=np.int64); im = np.zeros(n, dtype=np.int64)
        re[0] = 1 << 20
        gr, gi = golden.golden_fft(re, im)
        # FFT(delta)/N = amplitude/N everywhere
        np.testing.assert_allclose(gr, (1 << 20) // n, atol=1)
        np.testing.assert_allclose(gi, 0, atol=1)

    def test_ifft_inverts_fft(self) -> None:
        n = 1024
        re, im = golden.vector_input(seed=2, n=n)
        fr, fi = golden.golden_fft(re, im)
        rr, ri = golden.golden_ifft(fr, fi)
        # Round trip = x/N (block exponent +log2 N); compare to rhe(x, log2 N).
        s = int(np.log2(n))
        exp_r = golden.rhe(re, s)
        exp_i = golden.rhe(im, s)
        self.assertLess(np.max(np.abs(rr - exp_r)), 40)
        self.assertLess(np.max(np.abs(ri - exp_i)), 40)

    def test_deterministic(self) -> None:
        re, im = golden.vector_input(seed=3, n=512)
        a = golden.golden_fft(re, im)
        b = golden.golden_fft(re, im)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


class ModReluInt(unittest.TestCase):
    def test_zero_threshold_is_identity(self) -> None:
        re, im = golden.vector_input(seed=4, n=512)
        yr, yi = golden.modrelu_int(re, im, b_q23=0, block_exp=0)
        np.testing.assert_array_equal(yr, re)
        np.testing.assert_array_equal(yi, im)

    def test_threshold_kills_small_samples(self) -> None:
        zr = np.array([100, 200000], dtype=np.int64)
        zi = np.zeros(2, dtype=np.int64)
        yr, _ = golden.modrelu_int(zr, zi, b_q23=-1000, block_exp=0)
        self.assertEqual(yr[0], 0)          # 100 < 1000 threshold -> gated
        self.assertGreater(yr[1], 190000)   # large sample survives, shrunk by b

    def test_exponent_compensation(self) -> None:
        # Same signal at two block exponents: gate decisions must match when the
        # threshold is compensated.
        zr = np.array([500, 5000, 50000], dtype=np.int64)
        zi = np.zeros(3, dtype=np.int64)
        y0, _ = golden.modrelu_int(zr, zi, b_q23=-2000, block_exp=0)
        y2, _ = golden.modrelu_int(zr >> 2, zi, b_q23=-2000, block_exp=2)
        np.testing.assert_array_equal(y0 > 0, y2 > 0)


class PinnedVectors(unittest.TestCase):
    """The cross-implementation contract: C and RTL must reproduce these."""

    def _compute(self, kind: str, seed: int, n: int) -> str:
        re, im = golden.vector_input(seed=seed, n=n)
        if kind == "fft":
            yr, yi = golden.golden_fft(re, im)
        else:
            fr, fi = golden.golden_fft(re, im)
            hr = np.full(n, 1 << 14, dtype=np.int64)  # H = 0.5 flat table
            hi = np.zeros(n, dtype=np.int64)
            mr, mi = golden.spectral_multiply(fr, fi, hr, hi)
            yr, yi = golden.golden_ifft(mr, mi)
        return golden.digest(yr, yi)

    def test_pinned_digests(self) -> None:
        pins = {
            ("fft", 11, 256): "ccea7a8301f8b8372bb1d2365b4e06e7330200ffee249e0f1634210fb9dd7a22",
            ("fft", 12, 1024): "09ea67d3e313581054387508d4709b2dc6b04cdc26d2a5036dd024a4037b82b1",
            ("fft", 13, 4096): "481b10ecbe9823d42e3e279ca88f76aba2e7e2e7229926cb3bd81edfc8f8aa80",
            ("roundtrip", 21, 1024): "cd276c1485d97ce30b707aa9a9cd08d4c6f8170333020e0413f740a29bfcc0d2",
        }
        for (kind, seed, n), pin in pins.items():
            got = self._compute(kind, seed, n)
            self.assertEqual(got, pin, msg=f"vector ({kind},{seed},{n}) diverged")


if __name__ == "__main__":
    unittest.main()
