"""Fidelity to the clean waveform (the core truth) and blind recovery (the proxy).

Two families of metric, and the relationship between them is the whole point:

- **Coherence** (:func:`coherence`) is the core truth. It is the fraction of the
  operator output that is a genuine copy of the clean transmitted waveform, after
  maximizing over exactly the transformations a coherent receiver removes for
  free: complex gain (carrier phase + amplitude), timing (integer delay), and
  carrier frequency offset. What remains -- intersymbol interference, noise,
  distortion -- is genuine signal quality. This single quantity is, by
  construction, invariant to gain inflation (the gain is fitted out), punishing
  of content collapse (an output orthogonal to the truth has coherence zero), and
  self-regularizing (any added out-of-band energy raises the output norm without
  raising the correlation, so coherence falls). No penalty, deadzone, clip, or
  lock-gate is needed around it. Available wherever ground truth exists: sim and
  twin.

- **Blind recovery** (:func:`blind_recover`) is a proxy for the core truth, for
  the hardware regime where the clean waveform is unavailable. It is a
  fractionally-spaced CMA equalizer with a convergence indicator and a blind
  modulus-dispersion ISI proxy. It is trusted only after ``reward.proxy_validity``
  certifies in sim that it tracks the coherence reward; parity with the classifier's
  TypeScript reference is a separate blocking golden gate.

Everything operates per stream; callers average over the channel dimension.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# the core truth: coherence to the clean waveform
# ---------------------------------------------------------------------------
def _shift(x: np.ndarray, d: int) -> np.ndarray:
    """A true (non-circular) integer shift: vacated samples are zeroed."""
    y = np.roll(x, d)
    if d > 0:
        y[:d] = 0.0
    elif d < 0:
        y[d:] = 0.0
    return y


def _estimate_cfo_bins(z: np.ndarray, ref: np.ndarray) -> float:
    """Carrier frequency offset (in DFT bins) between ``z`` and ``ref``.

    The cross-product ``z * conj(ref)`` has a spectral line at the offset; its
    peak, refined by parabolic interpolation for sub-bin resolution, is the CFO.
    """
    w = z * np.conj(ref)
    mag = np.abs(np.fft.fft(w))
    n = mag.size
    k = int(np.argmax(mag))
    a, b, c = mag[(k - 1) % n], mag[k], mag[(k + 1) % n]
    denom = a - 2 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    freq = k + delta
    return freq - n if freq > n / 2 else freq


def coherence(estimate: np.ndarray, reference: np.ndarray, max_delay: int = 24) -> float:
    """Coherence ``gamma^2`` in [0, 1] to ``reference``, maximized over the
    coherent-receiver nuisance group (gain, phase, delay, carrier frequency).

    ``1`` means the estimate is exactly a gain/phase/delay/CFO transform of the
    reference (perfect fidelity); ``0`` means orthogonal (no signal). This is the
    core signal-quality quantity; the reward is its improvement.
    """
    z = np.asarray(estimate, dtype=np.complex128)
    x = np.asarray(reference, dtype=np.complex128)
    zz = float(np.vdot(z, z).real)
    if zz <= 0.0 or x.size == 0:
        return 0.0

    def best_over_delay(xr: np.ndarray) -> tuple[float, int]:
        best, best_d = 0.0, 0
        for d in range(-max_delay, max_delay + 1):
            xd = _shift(xr, d)
            xx = float(np.vdot(xd, xd).real)
            if xx <= 0.0:
                continue
            g2 = float(np.abs(np.vdot(xd, z)) ** 2 / (xx * zz))
            if g2 > best:
                best, best_d = g2, d
        return best, best_d

    # Coarse alignment at zero CFO, then estimate and remove the CFO, then refine.
    coarse, d0 = best_over_delay(x)
    n = z.size
    freq = _estimate_cfo_bins(z, _shift(x, d0))
    xr = x * np.exp(2j * np.pi * freq * np.arange(n) / n)
    refined, _ = best_over_delay(xr)
    return float(min(max(coarse, refined), 1.0))


def alignment_error(estimate: np.ndarray, reference: np.ndarray, max_delay: int = 24) -> float:
    """``1 - coherence``: the fraction of the estimate not explained by the truth.

    Retained as the complementary view of :func:`coherence`; ``0`` is a perfect
    match, ``1`` uncorrelated.
    """
    return 1.0 - coherence(estimate, reference, max_delay)


# ---------------------------------------------------------------------------
# blind recovery
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecoveryResult:
    """Output of blind recovery on one stream."""

    equalized: np.ndarray  # (n_symbols,) equalized symbol estimates
    residual_isi: float    # blind modulus-dispersion proxy (>=0, lower is better)
    converged: bool        # CMA cost stabilized below the lock threshold
    snr_db: float          # blind PSD-floor SNR estimate

    # The lock, per the corrected design: residual ISI under threshold AND the
    # convergence indicator set. ``snr_db`` is a pre-equalizer PSD statistic and
    # is not used to gate the lock.
    def locked(self, isi_threshold: float = 0.22) -> bool:
        return self.converged and self.residual_isi < isi_threshold


def blind_snr_db(x: np.ndarray) -> float:
    """Blind SNR estimate from the power spectrum: peak-band vs noise floor."""
    x = np.asarray(x, dtype=np.complex128)
    if x.size == 0:
        return -np.inf
    psd = np.abs(np.fft.fft(x)) ** 2
    psd_sorted = np.sort(psd)
    floor = float(np.mean(psd_sorted[: max(1, psd.size // 4)]))  # lowest quartile
    total = float(np.mean(psd))
    if floor <= 0.0:
        return 60.0
    signal = max(total - floor, floor * 1e-6)
    return float(10.0 * np.log10(signal / floor))


def blind_recover(
    observed: np.ndarray,
    sps: int,
    taps_per_symbol: int = 2,
    mu: float = 3e-3,
    max_passes: int = 4,
) -> RecoveryResult:
    """Fractionally-spaced CMA equalizer with a convergence indicator.

    The equalizer is fractionally spaced (``taps_per_symbol`` taps per symbol
    period) and decimates to one output per symbol. The Godard radius targets a
    unit-power constellation. Convergence is declared when the CMA cost over the
    final pass is both low and stable.
    """
    x = np.asarray(observed, dtype=np.complex128).ravel()
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    n_taps = int(taps_per_symbol * sps) | 1  # odd, spans a couple symbols
    w = np.zeros(n_taps, dtype=np.complex128)
    w[n_taps // 2] = 1.0  # center-spike init
    radius = 1.0  # E[|s|^4]/E[|s|^2] for a unit-power constellation ~ 1
    n = x.size
    stride = 1  # fractionally spaced: slide one sample, decimate outputs by sps
    costs = []
    equalized_last: list[complex] = []
    for p in range(max_passes):
        pass_cost = 0.0
        count = 0
        equalized_last = []
        for center in range(n_taps // 2, n - n_taps // 2, sps // 2 if sps >= 2 else 1):
            window = x[center - n_taps // 2 : center + n_taps // 2 + 1]
            if window.size != n_taps:
                continue
            y = np.vdot(w[::-1].conj(), window)  # w^T window
            err = y * (np.abs(y) ** 2 - radius)
            w = w - mu * err * np.conj(window)[::-1]
            pass_cost += float((np.abs(y) ** 2 - radius) ** 2)
            count += 1
            if p == max_passes - 1 and (center - n_taps // 2) % sps == 0:
                equalized_last.append(y)
        costs.append(pass_cost / max(count, 1))
    equalized = np.array(equalized_last, dtype=np.complex128)
    # Blind ISI proxy: normalized dispersion of the equalized modulus.
    if equalized.size:
        power = np.mean(np.abs(equalized) ** 2)
        equalized_n = equalized / (np.sqrt(power) + 1e-12)
        residual_isi = float(np.var(np.abs(equalized_n) ** 2))
    else:
        residual_isi = 1.0
    # Convergence: final cost is low and not worse than the previous pass.
    converged = bool(
        len(costs) >= 2 and costs[-1] < 0.5 and costs[-1] <= costs[-2] * 1.05
    )
    return RecoveryResult(
        equalized=equalized,
        residual_isi=residual_isi,
        converged=converged,
        snr_db=blind_snr_db(x),
    )
