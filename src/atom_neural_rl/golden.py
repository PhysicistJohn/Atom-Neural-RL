"""The golden integer arithmetic -- the single datapath all implementations match.

This module is the normative reference implementation of golden-arithmetic v1,
the arithmetic that the QEMU twin device and the FPGA RTL must reproduce
bit-for-bit. It exists to close the root blocker of the gap register: three
divergent arithmetics (float FFT here, Q2.30 CORDIC in the twin, 18-bit-twiddle
RTL contract) cannot form a bit-exactness chain. This one can, because every
operation is integer with a pinned rounding rule.

Pinned decisions (the spec document mirrors this docstring):

- **Data**: complex integers, 24-bit signed components, carried in int64.
  IQ16 input enters left-shifted by 8 (sign-extended), filling the 24-bit range.
- **Twiddles**: 18-bit signed Q1.17, from a committed ROM of 32768 (cos, sin)
  pairs covering theta in [0, pi) at 2pi/65536 resolution. For any N <= 2^16 the
  FFT twiddle W_N^k = (cos, -sin) at ROM index k * (65536/N). The ROM is
  generated once (round-half-even) and committed as little-endian int32 binary
  with a sha256 lock; implementations load it, never recompute it, so host libm
  can never perturb the chain.
- **Rounding**: ONE rule everywhere -- round-half-to-even on right shift
  (``rhe``). This deliberately replaces C's truncate-toward-zero and Verilog's
  floor shift, both of which must implement rhe explicitly; the C-vs-RTL
  negative-halving mismatch is thereby designed out rather than patched around.
- **FFT**: radix-2 DIT, bit-reversed input order, natural output order, one
  divide-by-2 (rhe) per stage: forward computes FFT(x)/N. Twiddle products are
  Q1.17 rounded back to data scale with ``rhe(v, 17)``.
- **IFFT**: conjugate -> forward -> conjugate, which equals the true IFFT
  exactly (the forward's 1/N is the IFFT's own 1/N). The spectral round trip
  IFFT(H . FFT(x)) therefore carries a net 1/N: the **block exponent** increases
  by log2(N) per round trip as metadata; sample containers stay 24-bit.
- **Spectral multiply**: tables are int16 Q1.15 mantissas with one shared int8
  exponent per table; ``Y = rhe(X * H_mant, 15)``, block exponent += table
  exponent.
- **Backbone** (lift / mixes / project): complex int16 Q1.15 matrix entries
  (round-half-even at compile); accumulate int64; ``rhe(acc, 15)`` per output.
- **modReLU**: magnitude by alpha-max-beta-min, m = rhe(15*max,4) + rhe(15*min,5)
  (alpha=15/16, beta=15/32, max error ~4%); threshold is a Q1.23 integer at
  block-exponent 0, shifted to the current block exponent before comparison
  (exponent compensation by construction); survivor scale s = ((m+b)<<15)//m
  (floor division of nonnegative operands -- identical in C and Verilog), output
  ``rhe(z*s, 15)``. b=0 gives s=2^15 and exact identity.

Everything here is deliberately scalar-simple in structure (vectorized in numpy
for speed, but stage-by-stage, butterfly-by-butterfly in semantics) so the C and
Verilog implementations are line-for-line transliterations.
"""
from __future__ import annotations

import hashlib
import os
from typing import Tuple

import numpy as np

DATA_BITS = 24
DATA_MAX = (1 << (DATA_BITS - 1)) - 1
DATA_MIN = -(1 << (DATA_BITS - 1))
TWIDDLE_FRAC = 17          # Q1.17
TABLE_FRAC = 15            # Q1.15 spectral tables / backbone
ROM_HALF_TURN = 32768      # entries covering [0, pi)
ROM_RESOLUTION = 65536     # angle unit = 2*pi / 65536
LOG2N_MAX = 16

_ROM_PATH = os.path.join(os.path.dirname(__file__), "data", "twiddle-rom-q117.bin")


# ---------------------------------------------------------------------------
# rounding -- the one rule
# ---------------------------------------------------------------------------
def rhe(v, s: int):
    """Round-half-to-even arithmetic right shift by ``s`` (vectorized, int64).

    Defined for all integers, positive and negative, with ties going to the
    even result. This is THE rounding rule of golden-arithmetic v1; C and RTL
    implement this exact function.
    """
    if s == 0:
        return np.asarray(v, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    half = np.int64(1) << (s - 1)
    mask = (np.int64(1) << s) - 1
    q = (v + half) >> s
    # ties: remainder exactly half and quotient odd -> step back to even
    tie = (v & mask) == half
    q = q - (tie & ((q & 1) == 1)).astype(np.int64)
    return q


def clamp24(v):
    return np.clip(np.asarray(v, dtype=np.int64), DATA_MIN, DATA_MAX)


# ---------------------------------------------------------------------------
# the committed twiddle ROM
# ---------------------------------------------------------------------------
def generate_twiddle_rom() -> np.ndarray:
    """Generate the ROM (int32 interleaved cos,sin). Used ONCE to create the
    committed artifact; consumers load the committed file."""
    k = np.arange(ROM_HALF_TURN, dtype=np.float64)
    theta = 2.0 * np.pi * k / ROM_RESOLUTION
    scale = float(1 << TWIDDLE_FRAC)
    limit = (1 << TWIDDLE_FRAC) - 1

    def q(x: np.ndarray) -> np.ndarray:
        # round-half-even then clamp so +1.0 maps to the max representable
        r = np.rint(x * scale).astype(np.int64)  # rint is round-half-even
        return np.clip(r, -limit - 1, limit).astype(np.int32)

    rom = np.empty(ROM_HALF_TURN * 2, dtype=np.int32)
    rom[0::2] = q(np.cos(theta))
    rom[1::2] = q(np.sin(theta))
    return rom


def load_twiddle_rom() -> np.ndarray:
    """Load the committed ROM (int32 LE interleaved cos,sin pairs)."""
    data = np.fromfile(_ROM_PATH, dtype="<i4")
    if data.size != ROM_HALF_TURN * 2:
        raise ValueError("twiddle ROM has wrong size")
    return data


def rom_sha256() -> str:
    with open(_ROM_PATH, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


_ROM_CACHE: np.ndarray = None


def _rom() -> np.ndarray:
    global _ROM_CACHE
    if _ROM_CACHE is None:
        _ROM_CACHE = load_twiddle_rom()
    return _ROM_CACHE


def twiddles_for(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """FFT twiddles W_N^k = (cos, -sin) for k in [0, N/2), from the ROM."""
    stride = ROM_RESOLUTION // n
    idx = np.arange(n // 2) * stride
    rom = _rom()
    wr = rom[2 * idx].astype(np.int64)
    wi = -rom[2 * idx + 1].astype(np.int64)  # e^{-j theta}
    return wr, wi


# ---------------------------------------------------------------------------
# the golden FFT (radix-2 DIT, /2 per stage, rhe everywhere)
# ---------------------------------------------------------------------------
def bit_reverse_indices(n: int) -> np.ndarray:
    bits = int(np.log2(n))
    idx = np.arange(n)
    rev = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def golden_fft(re: np.ndarray, im: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Forward golden FFT of 24-bit integer samples. Returns FFT(x)/N (24-bit).

    Input in natural order (the bit-reversal permutation is applied inside);
    output in natural order.
    """
    n = re.size
    if n & (n - 1):
        raise ValueError("N must be a power of two")
    rev = bit_reverse_indices(n)
    xr = np.asarray(re, dtype=np.int64)[rev].copy()
    xi = np.asarray(im, dtype=np.int64)[rev].copy()
    stages = int(np.log2(n))
    for s in range(stages):
        half = 1 << s          # butterflies per group half-size
        step = half << 1       # group size
        # twiddle for position j in [0, half): W_N^{j * (N/step)}
        wr_full, wi_full = twiddles_for(n)
        tw_idx = (np.arange(half) * (n // step))
        wr = wr_full[tw_idx]
        wi = wi_full[tw_idx]
        # vectorized over groups
        starts = np.arange(0, n, step)
        ia = (starts[:, None] + np.arange(half)[None, :]).ravel()
        ib = ia + half
        br, bi_ = xr[ib], xi[ib]
        wrb = np.tile(wr, starts.size)
        wib = np.tile(wi, starts.size)
        # twiddle product, Q1.17 -> data scale
        tr = rhe(br * wrb - bi_ * wib, TWIDDLE_FRAC)
        ti = rhe(br * wib + bi_ * wrb, TWIDDLE_FRAC)
        ar, ai = xr[ia], xi[ia]
        # butterfly with /2 (rhe) per stage
        xr[ia] = rhe(ar + tr, 1)
        xi[ia] = rhe(ai + ti, 1)
        xr[ib] = rhe(ar - tr, 1)
        xi[ib] = rhe(ai - ti, 1)
    return clamp24(xr), clamp24(xi)


def golden_ifft(re: np.ndarray, im: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse via conjugate -> forward -> conjugate. Equals true IFFT exactly."""
    yr, yi = golden_fft(np.asarray(re, dtype=np.int64), -np.asarray(im, dtype=np.int64))
    return yr, -yi


# ---------------------------------------------------------------------------
# spectral multiply, backbone, modReLU
# ---------------------------------------------------------------------------
def spectral_multiply(xr, xi, hr_q15, hi_q15) -> Tuple[np.ndarray, np.ndarray]:
    """Per-bin complex multiply by a Q1.15 table (mantissas int16); rhe(.,15)."""
    xr = np.asarray(xr, dtype=np.int64); xi = np.asarray(xi, dtype=np.int64)
    hr = np.asarray(hr_q15, dtype=np.int64); hi = np.asarray(hi_q15, dtype=np.int64)
    yr = rhe(xr * hr - xi * hi, TABLE_FRAC)
    yi = rhe(xr * hi + xi * hr, TABLE_FRAC)
    return clamp24(yr), clamp24(yi)


def backbone_apply(matrix_q15: np.ndarray, vr: np.ndarray, vi: np.ndarray):
    """Apply a complex Q1.15 integer matrix (int16 mantissas as int32 complex
    pairs) to a stack of W channel streams. matrix is (out, in, 2) int64
    [re,im] mantissas. Returns (out, N) int64 pair."""
    out_dim, in_dim, _ = matrix_q15.shape
    n = vr.shape[1]
    yr = np.zeros((out_dim, n), dtype=np.int64)
    yi = np.zeros((out_dim, n), dtype=np.int64)
    for o in range(out_dim):
        acc_r = np.zeros(n, dtype=np.int64)
        acc_i = np.zeros(n, dtype=np.int64)
        for c in range(in_dim):
            mr = matrix_q15[o, c, 0]; mi = matrix_q15[o, c, 1]
            acc_r += vr[c] * mr - vi[c] * mi
            acc_i += vr[c] * mi + vi[c] * mr
        yr[o] = clamp24(rhe(acc_r, TABLE_FRAC))
        yi[o] = clamp24(rhe(acc_i, TABLE_FRAC))
    return yr, yi


def modrelu_int(zr, zi, b_q23: int, block_exp: int) -> Tuple[np.ndarray, np.ndarray]:
    """Integer modReLU with exponent-compensated threshold.

    ``b_q23`` is the (non-positive) threshold as a Q1.23 integer at block
    exponent 0; it is shifted to the current block exponent before comparison so
    the gate decision is independent of block scaling.
    """
    zr = np.asarray(zr, dtype=np.int64); zi = np.asarray(zi, dtype=np.int64)
    ar = np.abs(zr); ai = np.abs(zi)
    mx = np.maximum(ar, ai); mn = np.minimum(ar, ai)
    mag = rhe(15 * mx, 4) + rhe(15 * mn, 5)         # alpha-max-beta-min
    # exponent compensation: shift threshold into current block scale
    if block_exp >= 0:
        b_eff = rhe(np.int64(b_q23), block_exp) if block_exp > 0 else np.int64(b_q23)
    else:
        b_eff = np.int64(b_q23) << (-block_exp)
    keep = mag + b_eff
    live = (keep > 0) & (mag > 0)
    scale = np.zeros_like(mag)
    scale[live] = (keep[live] << TABLE_FRAC) // mag[live]   # floor, nonneg/nonneg
    yr = rhe(zr * scale, TABLE_FRAC)
    yi = rhe(zi * scale, TABLE_FRAC)
    return clamp24(yr), clamp24(yi)


# ---------------------------------------------------------------------------
# deterministic vector generation (pinned PRNG -- identical in C and Verilog)
# ---------------------------------------------------------------------------
def splitmix64(seed: int, count: int) -> np.ndarray:
    """The pinned PRNG of golden-arithmetic v1 (for test vectors only)."""
    out = np.empty(count, dtype=np.uint64)
    x = np.uint64(seed)
    GAMMA = np.uint64(0x9E3779B97F4A7C15)
    M1 = np.uint64(0xBF58476D1CE4E5B9)
    M2 = np.uint64(0x94D049BB133111EB)
    with np.errstate(over="ignore"):
        for i in range(count):
            x = x + GAMMA
            z = x
            z = (z ^ (z >> np.uint64(30))) * M1
            z = (z ^ (z >> np.uint64(27))) * M2
            z = z ^ (z >> np.uint64(31))
            out[i] = z
    return out


def vector_input(seed: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic IQ16-range test input, sign-extended <<8 to 24-bit."""
    words = splitmix64(seed, n)
    re16 = (words & np.uint64(0xFFFF)).astype(np.int64)
    im16 = ((words >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.int64)
    re16 = np.where(re16 >= 32768, re16 - 65536, re16)
    im16 = np.where(im16 >= 32768, im16 - 65536, im16)
    return re16 << 8, im16 << 8


def digest(*arrays) -> str:
    """sha256 over int32-LE serialization of the given int arrays."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.asarray(a, dtype=np.int64).astype("<i4").tobytes())
    return h.hexdigest()
