"""Deterministic linearly-modulated waveform synthesis (the gym's signal source).

A compact, dependency-free stand-in for the SignalLab reference generator: RRC
pulse-shaped PSK/QAM at a chosen samples-per-symbol, seeded and bit-frozen. It
produces the clean complex baseband the channel model then impairs, plus the
transmitted symbols so truth-based metrics (EVM) are available where ground
truth exists.

All arrays are complex baseband, shape ``(N,)`` for a single stream; the gym
stacks streams into the ``(channels, N)`` convention used everywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# Reference constellations (unit average energy), mirroring the shipped
# Atomizer/SignalLab reference set (BPSK excluded, as in that work).
_CONSTELLATIONS: Dict[str, np.ndarray] = {}


def _register_psk(name: str, order: int) -> None:
    k = np.arange(order)
    _CONSTELLATIONS[name] = np.exp(1j * 2 * np.pi * k / order)


def _register_qam(name: str, side: int) -> None:
    levels = np.arange(-(side - 1), side, 2)
    re, im = np.meshgrid(levels, levels)
    points = (re + 1j * im).ravel().astype(np.complex128)
    points /= np.sqrt(np.mean(np.abs(points) ** 2))
    _CONSTELLATIONS[name] = points


_register_psk("qpsk", 4)
_register_psk("8psk", 8)
_register_qam("16qam", 4)
_register_qam("64qam", 8)
_register_qam("256qam", 16)

MODULATIONS = tuple(_CONSTELLATIONS.keys())


@dataclass(frozen=True)
class WaveformProfile:
    """A linearly-modulated emitter: modulation, samples/symbol, RRC roll-off."""

    modulation: str
    sps: int
    rolloff: float = 0.35
    rrc_span: int = 8

    def __post_init__(self) -> None:
        if self.modulation not in _CONSTELLATIONS:
            raise ValueError(f"unknown modulation {self.modulation!r}")
        if self.sps < 2:
            raise ValueError("sps must be >= 2 (occupied bandwidth < Nyquist)")
        if not 0.0 < self.rolloff <= 1.0:
            raise ValueError("rolloff must be in (0, 1]")

    def occupied_bandwidth_fraction(self) -> float:
        """Occupied bandwidth as a fraction of the sample rate: ``(1+beta)/sps``."""
        return (1.0 + self.rolloff) / self.sps


def rrc_taps(sps: int, rolloff: float, span: int) -> np.ndarray:
    """Root-raised-cosine taps, unit-energy normalized, length ``span*sps + 1``."""
    n = span * sps
    t = (np.arange(n + 1) - n / 2) / sps
    beta = rolloff
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(4.0 * beta * ti) - 1.0) < 1e-9:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den
    taps /= np.sqrt(np.sum(taps ** 2))
    return taps


@dataclass(frozen=True)
class SynthResult:
    """One synthesized clean capture plus the ground truth it came from."""

    iq: np.ndarray        # (N,) clean complex baseband
    symbols: np.ndarray   # (n_symbols,) transmitted constellation points
    sps: int
    profile: WaveformProfile


def synthesize(profile: WaveformProfile, n_samples: int, seed: int) -> SynthResult:
    """Synthesize ``n_samples`` of clean RRC-shaped modulation, deterministically."""
    rng = np.random.default_rng(seed)
    taps = rrc_taps(profile.sps, profile.rolloff, profile.rrc_span)
    constellation = _CONSTELLATIONS[profile.modulation]
    n_symbols = int(np.ceil(n_samples / profile.sps)) + profile.rrc_span + 2
    idx = rng.integers(0, constellation.size, size=n_symbols)
    symbols = constellation[idx]
    # Upsample: place symbols every sps samples, zero-fill, then pulse-shape.
    upsampled = np.zeros(n_symbols * profile.sps, dtype=np.complex128)
    upsampled[:: profile.sps] = symbols
    shaped = np.convolve(upsampled, taps, mode="full")
    # Drop the filter transient and take the requested length.
    start = (taps.size - 1) // 2
    iq = shaped[start : start + n_samples]
    # Normalize to unit average power.
    iq = iq / np.sqrt(np.mean(np.abs(iq) ** 2))
    return SynthResult(iq=iq, symbols=symbols, sps=profile.sps, profile=profile)
