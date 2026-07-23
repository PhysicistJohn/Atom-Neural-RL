"""AD9361 impairment measurement: the sim2real harness for the arriving board.

Estimators for the board-reality impairments the gym does not yet model, each a
pure function of captured IQ so they run identically on sim stand-ins today and
real loopback captures on day one. The calibration waveform is a known
reference (transmitted over cabled/attenuated loopback or internal BIST), which
restores truth-based measurement -- the reliable path, per the measured finding
that blind over-the-air adaptation is untrustworthy.

Measured values feed ChannelParams / gym extensions so pretraining matches the
actual radio. AGC must be pinned to manual gain (MGC) during characterization;
every estimate here is per-gain-index.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def calibration_waveform(n: int, tone_bins: tuple = (5, 17, 41), amplitude: float = 0.3) -> np.ndarray:
    """A known multitone calibration waveform (complex, unit-safe amplitude).

    Deterministic multitone with coprime bin spacing: distinguishes linear
    response (tones), DC offset (bin 0), IQ imbalance (image bins), and noise
    (everything else). No randomness -- byte-identical everywhere.
    """
    t = np.arange(n)
    x = np.zeros(n, dtype=np.complex128)
    for k in tone_bins:
        x += np.exp(2j * np.pi * k * t / n)
    return amplitude * x / len(tone_bins)


@dataclass(frozen=True)
class ImpairmentEstimate:
    dc_offset: complex          # additive DC (LO leakage at baseband)
    iq_gain_imbalance_db: float # I/Q amplitude imbalance
    iq_phase_error_deg: float   # quadrature phase error
    cfo_bins: float             # carrier frequency offset in DFT bins
    noise_floor_dbfs: float     # mean noise PSD outside tones/images/DC


def estimate_impairments(
    captured: np.ndarray, reference: np.ndarray, tone_bins: tuple = (5, 17, 41)
) -> ImpairmentEstimate:
    """Estimate DC, IQ imbalance, CFO, and noise floor from a loopback capture.

    ``captured`` is the received block, ``reference`` the transmitted
    calibration waveform (same length). IQ imbalance is read from the image-tone
    energy: gain/phase imbalance maps tone k into conjugate image -k with
    complex ratio K = (1 - g e^{j phi}) / (1 + g e^{j phi}).
    """
    z = np.asarray(captured, dtype=np.complex128)
    n = z.size
    # DC first, as the raw capture mean: the calibration tones integrate to zero
    # over whole periods and noise averages out, so the mean IS the additive DC,
    # robust to CFO (which is applied before DC in the physical chain). Remove it
    # before the CFO and imbalance fits so it cannot smear across bins.
    dc = complex(np.mean(z))
    z = z - dc
    # CFO: data-aided, robust to IQ imbalance. Integer bin from the
    # cross-spectrum peak; the fractional part maximizes DIRECT-tone energy after
    # de-rotation. A phase-slope estimate is biased by the imbalance image tones;
    # maximizing direct-tone energy is not, since the images never land on the
    # reference bins.
    t = np.arange(n)
    w = z * np.conj(np.asarray(reference, dtype=np.complex128))
    k = int(np.argmax(np.abs(np.fft.fft(w))))
    k_signed = k - n if k > n // 2 else k
    tones = list(tone_bins)

    def direct_energy(f: float) -> float:
        Zf = np.fft.fft(z * np.exp(-2j * np.pi * f * t / n))
        return float(np.sum(np.abs(Zf[tones]) ** 2))

    grid = np.linspace(k_signed - 1.0, k_signed + 1.0, 401)
    cfo = float(grid[int(np.argmax([direct_energy(f) for f in grid]))])
    z = z * np.exp(-2j * np.pi * cfo * t / n)

    Z = np.fft.fft(z) / n

    # IQ imbalance from tone/image ratios. For y = mu z + nu conj(z) with a real
    # reference tone at +k, Y[k] = mu, Y[-k] = nu, so K := Y[-k]/Y[k] = nu/mu =
    # (1 - ge)/(1 + ge), inverted directly as ge = (1 - K)/(1 + K).
    Ks = []
    for kb in tone_bins:
        direct = Z[kb]
        image = Z[(-kb) % n]
        if abs(direct) > 0:
            Ks.append(image / direct)
    K = complex(np.mean(Ks)) if Ks else 0.0 + 0j
    ge = (1 - K) / (1 + K)
    gain_db = float(20 * np.log10(np.abs(ge))) if np.abs(ge) > 0 else 0.0
    phase_deg = float(np.degrees(np.angle(ge)))

    # Noise floor: PSD excluding DC, tones, images (and adjacent bins).
    excluded = {0}
    for kb in tone_bins:
        for off in (-1, 0, 1):
            excluded.add((kb + off) % n)
            excluded.add((-kb + off) % n)
    keep = np.array([i for i in range(n) if i not in excluded])
    noise_power = float(np.mean(np.abs(Z[keep]) ** 2)) * n  # per-sample power
    noise_dbfs = 10 * np.log10(noise_power + 1e-30)

    return ImpairmentEstimate(
        dc_offset=dc,
        iq_gain_imbalance_db=gain_db,
        iq_phase_error_deg=phase_deg,
        cfo_bins=float(cfo),
        noise_floor_dbfs=noise_dbfs,
    )


def apply_impairments(
    clean: np.ndarray,
    dc_offset: complex = 0.0,
    iq_gain_imbalance_db: float = 0.0,
    iq_phase_error_deg: float = 0.0,
    cfo_bins: float = 0.0,
    noise_dbfs: float = -np.inf,
    seed: int = 0,
) -> np.ndarray:
    """The forward model (for tests and for extending the gym with measured
    values): applies the same impairments the estimator measures."""
    z = np.asarray(clean, dtype=np.complex128).copy()
    n = z.size
    g = 10 ** (iq_gain_imbalance_db / 20)
    phi = np.radians(iq_phase_error_deg)
    ge = g * np.exp(1j * phi)
    # standard IQ-imbalance model: y = mu*z + nu*conj(z)
    mu = 0.5 * (1 + ge)
    nu = 0.5 * (1 - ge)
    z = mu * z + nu * np.conj(z)
    if cfo_bins:
        z = z * np.exp(2j * np.pi * cfo_bins * np.arange(n) / n)
    z = z + dc_offset
    if np.isfinite(noise_dbfs):
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(10 ** (noise_dbfs / 10) / 2)
        z = z + sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return z
