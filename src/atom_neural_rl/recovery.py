"""Blind recovery and fidelity metrics -- the reward judges.

Two families of metric, matching the design:

- **Truth-anchored fidelity** (:func:`alignment_error`): the normalized residual
  after fitting the single complex scale and integer delay that best match an
  estimate to the known clean waveform. This is the mandatory data-fidelity
  anchor. It is immune to gain inflation (the scale is fitted out) and it
  punishes content collapse (notching signal away increases the residual toward
  the clean truth). Available wherever ground truth exists: sim and twin.

- **Blind recovery** (:func:`blind_recover`): a fractionally-spaced CMA equalizer
  with a convergence indicator and a blind modulus-dispersion ISI proxy and a
  PSD-floor SNR estimate. This is the judge that works on real captures with no
  truth. It is a Python port of the classifier's recovery path; parity with the
  TypeScript reference is a separate blocking golden gate (``scripts/ts_parity``)
  and is not asserted numerically here.

Everything operates per stream; callers average over the channel dimension.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# truth-anchored fidelity
# ---------------------------------------------------------------------------
def alignment_error(estimate: np.ndarray, reference: np.ndarray, max_delay: int = 24) -> float:
    """Normalized residual after the best complex-scale + integer-delay fit.

    ``0`` is a perfect match to ``reference`` up to gain/phase/delay; ``~1`` is
    uncorrelated. Gain- and phase-invariant by construction.
    """
    estimate = np.asarray(estimate, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    n = estimate.size
    est_energy = float(np.vdot(estimate, estimate).real)
    if est_energy <= 0.0:
        return 1.0
    best = 1.0
    for d in range(-max_delay, max_delay + 1):
        ref = np.roll(reference, d)
        # zero the wrapped region so the delay is a true shift, not circular
        if d > 0:
            ref[:d] = 0.0
        elif d < 0:
            ref[d:] = 0.0
        ref_energy = float(np.vdot(ref, ref).real)
        if ref_energy <= 0.0:
            continue
        a = np.vdot(ref, estimate) / ref_energy  # optimal complex scale
        residual = estimate - a * ref
        err = float(np.vdot(residual, residual).real) / est_energy
        best = min(best, err)
    return best


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
