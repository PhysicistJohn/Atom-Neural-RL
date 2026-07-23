"""Fully-integer operator execution on the golden arithmetic.

This replaces :class:`execution.AbiExecutor` as the bit-exactness reference: the
FFT/IFFT here are the golden integer kernels (18-bit ROM twiddles, rhe rounding,
/2 per stage), not numpy's float FFT, so this executor IS the arithmetic the
QEMU twin core and the RTL reproduce bit-for-bit. ``AbiExecutor`` remains as the
faster float-FFT approximation for training throughput; promotion evidence and
cross-implementation vectors come from here.

Float <-> integer boundary, pinned:
- Input complex floats are scaled by 2^23 (unit float == 24-bit full scale)
  with round-half-even, matching IQ16<<8 for +-1.0-normalized captures.
- Kernel tables: the rational kernel is evaluated in float on the N-point grid
  (the ARM's job in the real system), then quantized round-half-even to Q1.15
  mantissas with a shared power-of-two exponent -- the exact bank table format.
- Output returns to float as value * 2^(block_exp) / 2^23.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from . import golden
from .operator import NeuralOperator, OperatorConfig

_UNIT = 1 << 23  # float 1.0 <-> 24-bit full scale


def _to_int(iq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    re = np.asarray(np.rint(iq.real * _UNIT), dtype=np.int64)
    im = np.asarray(np.rint(iq.imag * _UNIT), dtype=np.int64)
    return golden.clamp24(re), golden.clamp24(im)


def _table_q15(kernel, n: int, fs_hz: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Evaluate + quantize a kernel table: Q1.15 mantissas, shared exponent."""
    if fs_hz != kernel.anchor_hz:
        kernel = kernel.transport(fs_hz)
    h = kernel.response_on_grid(n)
    peak = float(np.max(np.abs(np.concatenate([h.real, h.imag])))) if h.size else 0.0
    exp = 0
    if peak > 0:
        while peak / (1 << exp) > 0.999969:  # fit in Q1.15
            exp += 1
    scale = (1 << golden.TABLE_FRAC) / (1 << exp)
    hr = np.asarray(np.rint(h.real * scale), dtype=np.int64)
    hi = np.asarray(np.rint(h.imag * scale), dtype=np.int64)
    lim = (1 << golden.TABLE_FRAC) - 1
    return np.clip(hr, -lim - 1, lim), np.clip(hi, -lim - 1, lim), exp


def _matrix_q15(m: np.ndarray) -> np.ndarray:
    out = np.empty((m.shape[0], m.shape[1], 2), dtype=np.int64)
    lim = (1 << golden.TABLE_FRAC) - 1
    out[:, :, 0] = np.clip(np.rint(m.real * (1 << golden.TABLE_FRAC)), -lim - 1, lim)
    out[:, :, 1] = np.clip(np.rint(m.imag * (1 << golden.TABLE_FRAC)), -lim - 1, lim)
    return out


class GoldenExecutor:
    """Bit-exact integer execution of an operator (the cross-impl reference)."""

    def __init__(self, operator: NeuralOperator) -> None:
        self.operator = operator
        self.config: OperatorConfig = operator.config

    # operator interface (so the same training loop can run through it)
    def adapted_vector(self) -> np.ndarray:
        return self.operator.adapted_vector()

    def with_adapted_vector(self, vector: np.ndarray) -> "GoldenExecutor":
        return GoldenExecutor(self.operator.with_adapted_vector(vector))

    def forward(self, iq: np.ndarray, fs_hz: float) -> np.ndarray:
        yr, yi, block_exp = self.forward_int(*_to_int(np.asarray(iq, dtype=np.complex128)), fs_hz)
        scale = float(2.0 ** block_exp) / _UNIT
        return (yr.astype(np.float64) + 1j * yi.astype(np.float64)) * scale

    def forward_int(
        self, re: np.ndarray, im: np.ndarray, fs_hz: float
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """The integer datapath: (channels, N) int24 in -> int24 out + block exp."""
        op = self.operator
        cfg = op.config
        n = re.shape[1]
        block_exp = 0
        s = int(np.log2(n))
        # lift
        vr, vi = golden.backbone_apply(_matrix_q15(op.backbone.lift), re, im)
        for layer in range(cfg.layers):
            sr = np.empty_like(vr); si = np.empty_like(vi)
            layer_exp = 0
            for w in range(cfg.width):
                hr, hi, hexp = _table_q15(op.kernels[layer][w], n, fs_hz)
                fr, fi = golden.golden_fft(vr[w], vi[w])
                mr, mi = golden.spectral_multiply(fr, fi, hr, hi)
                rr, ri = golden.golden_ifft(mr, mi)
                # renormalization: reinsert the round trip's 1/N as exponent
                sr[w] = rr; si[w] = ri
                layer_exp = max(layer_exp, hexp)
            block_exp += s + layer_exp
            mrx, mix = golden.backbone_apply(_matrix_q15(op.backbone.mixes[layer]), vr, vi)
            # mix path is at the pre-FFT scale; align it to the spectral path's
            # exponent by rhe-shifting down (the spectral path gained s+hexp).
            shift = s + layer_exp
            pr = sr + golden.rhe(mrx, shift)
            pi = si + golden.rhe(mix, shift)
            for w in range(cfg.width):
                b_q23 = int(np.rint(op.thresholds[layer, w] * _UNIT))
                vr[w], vi[w] = golden.modrelu_int(pr[w], pi[w], b_q23, block_exp)
        yr, yi = golden.backbone_apply(_matrix_q15(op.backbone.project), vr, vi)
        return yr, yi, block_exp
