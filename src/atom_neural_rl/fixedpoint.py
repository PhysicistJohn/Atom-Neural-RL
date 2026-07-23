"""Bit-exact fixed-point model of the operator datapath the fabric realizes.

Scope. The novel arithmetic the operator adds to the existing FFT engine is the
per-mode complex table multiply and the modReLU. Those are modelled here in the
exact fixed-point form the RTL and the twin device model will implement, so a
bank pretrained in float can be quantized once and its hardware behaviour
predicted deterministically:

- **Weight tables** are stored as int16 mantissas in a Q1.15 fraction with a
  single shared int8 base-2 exponent per table, so gains above unity are
  represented without losing the ~90 dB of per-entry dynamic range. This is what
  the bank compiler emits.
- **Spectral data** is block-floating: a 24-bit signed mantissa per component
  with one shared exponent per block, matching the DSP48E1 datapath width.
- **modReLU** compares magnitude against the threshold in exponent-compensated
  units, so the nonlinearity's decision is independent of the block exponent --
  the subtlety the verification flagged, pinned here so sim, twin, and RTL agree.

The shared FFT/IFFT core is pre-existing fabric; its own fixed-point behaviour is
the twin's contract (int32/Q2.30 CORDIC) and is out of scope for this model,
which covers exactly the operator-added datapath.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WEIGHT_FRAC_BITS = 15   # Q1.15 mantissa
DATA_BITS = 24          # block-float spectral mantissa width
_INT16_MAX = 2 ** 15 - 1
_INT16_MIN = -(2 ** 15)


@dataclass(frozen=True)
class SharedExpTable:
    """A complex table as int16 Q1.15 mantissas with one shared base-2 exponent."""

    mantissa_re: np.ndarray  # int16
    mantissa_im: np.ndarray  # int16
    exponent: int
    frac_bits: int = WEIGHT_FRAC_BITS

    def reconstruct(self) -> np.ndarray:
        scale = 2.0 ** (self.exponent - self.frac_bits)
        return (self.mantissa_re.astype(np.float64) + 1j * self.mantissa_im.astype(np.float64)) * scale

    @property
    def size(self) -> int:
        return int(self.mantissa_re.size)


def _shared_exponent(peak: float, frac_bits: int) -> int:
    """Smallest exponent ``e`` so that ``peak / 2^e`` fits in the Q mantissa."""
    if peak <= 0.0:
        return 0
    # need peak <= (2^frac_bits) * 2^(e - frac_bits) = 2^e  (mantissa < 2^frac_bits)
    # i.e. 2^e must exceed peak by the mantissa headroom; solve for integer e.
    e = int(np.ceil(np.log2(peak))) if peak > 1.0 else 0
    # Guarantee the largest mantissa is representable in int16.
    while round(peak * 2.0 ** (frac_bits - e)) > _INT16_MAX:
        e += 1
    return e


def quantize_table(table: np.ndarray, frac_bits: int = WEIGHT_FRAC_BITS) -> SharedExpTable:
    """Quantize a complex frequency-response table to shared-exponent Q1.15."""
    table = np.asarray(table, dtype=np.complex128)
    peak = float(np.max(np.abs(np.concatenate([table.real, table.imag])))) if table.size else 0.0
    exponent = _shared_exponent(peak, frac_bits)
    scale = 2.0 ** (frac_bits - exponent)
    re = np.clip(np.round(table.real * scale), _INT16_MIN, _INT16_MAX).astype(np.int16)
    im = np.clip(np.round(table.imag * scale), _INT16_MIN, _INT16_MAX).astype(np.int16)
    return SharedExpTable(re, im, exponent, frac_bits)


def round_block_float(x: np.ndarray, data_bits: int = DATA_BITS) -> np.ndarray:
    """Round complex data to a ``data_bits`` block-float mantissa (per-array exponent)."""
    x = np.asarray(x, dtype=np.complex128)
    peak = float(np.max(np.abs(np.concatenate([x.real, x.imag])))) if x.size else 0.0
    if peak == 0.0:
        return x.copy()
    exponent = int(np.ceil(np.log2(peak)))
    scale = 2.0 ** (data_bits - 1 - exponent)
    re = np.round(x.real * scale) / scale
    im = np.round(x.imag * scale) / scale
    return re + 1j * im


def fixed_point_spectral_multiply(
    table: SharedExpTable, spectrum: np.ndarray, data_bits: int = DATA_BITS
) -> np.ndarray:
    """``H_Q[k] * X[k]`` with the quantized table and block-float rounding of the product.

    Deterministic and identical to what the fabric multiply produces given the
    same table and input.
    """
    hq = table.reconstruct()
    product = hq * np.asarray(spectrum, dtype=np.complex128)
    return round_block_float(product, data_bits)


def modrelu_fixed(z: np.ndarray, b: float, data_bits: int = DATA_BITS) -> np.ndarray:
    """Exponent-compensated fixed-point modReLU.

    The magnitude and the threshold are compared in the same block-scaled units,
    so the gate is independent of the block exponent. Result is block-float
    rounded to ``data_bits``.
    """
    z = np.asarray(z, dtype=np.complex128)
    mag = np.abs(z)
    scale = np.maximum(0.0, mag + b)
    out = np.zeros_like(z)
    nz = mag > 0
    out[nz] = scale[nz] * (z[nz] / mag[nz])
    return round_block_float(out, data_bits)


def weight_quant_error_dbc(table: np.ndarray, frac_bits: int = WEIGHT_FRAC_BITS) -> float:
    """Relative table quantization error in dB (carrier-referenced, dBc)."""
    table = np.asarray(table, dtype=np.complex128)
    q = quantize_table(table, frac_bits).reconstruct()
    signal = float(np.mean(np.abs(table) ** 2))
    error = float(np.mean(np.abs(table - q) ** 2))
    if error == 0.0:
        return -np.inf
    return 10.0 * np.log10(error / signal) if signal > 0 else 0.0
