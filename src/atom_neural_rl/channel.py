"""Deterministic, seeded RF channel and receiver impairments.

The distortions the operator is meant to undo (multipath ISI) and the ones it
must be robust to (AWGN, carrier frequency offset, IQ imbalance) are applied
here as pure functions of a seed. Everything is reproducible: the same seed and
parameters give bit-identical output, which is what makes paired
candidate-vs-bypass evaluation and common-random-number CMA-ES sound.

Operates on ``(channels, N)`` complex arrays; per-channel impairment draws are
independent but derived from the one episode seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ChannelParams:
    """Impairment strengths for one episode.

    ``channel_seed``, when set, fixes the multipath realization independently of
    the episode seed, so a channel can be held constant while symbols and noise
    vary. This is exactly the hardware fine-tune regime: learn the inverse of a
    specific channel. When ``None``, the multipath is drawn from the episode seed
    (fresh channel per episode, the domain-randomization regime).
    """

    snr_db: float = 20.0
    multipath_taps: int = 3          # number of echo taps (>=1; 1 == no echo)
    multipath_spread: float = 0.4    # relative strength of echoes vs main tap
    cfo_cycles_per_block: float = 0.0  # carrier offset over the whole block
    iq_imbalance_db: float = 0.0     # gain imbalance between I and Q
    iq_phase_deg: float = 0.0        # quadrature phase error
    channel_seed: Optional[int] = None  # fix the multipath realization if set


def _multipath_filter(rng: np.random.Generator, params: ChannelParams, sps: int = 1) -> np.ndarray:
    """Symbol-spaced multipath filter: echoes at integer multiples of ``sps``.

    Sample-spaced echoes on an oversampled signal are a negligible fractional-
    symbol perturbation; real intersymbol interference lives at the symbol
    period. Echoes therefore sit at delays ``sps, 2*sps, ...`` with a mild decay,
    which is exactly the ISI a fractionally-spaced equalizer inverts.
    """
    n_echo = params.multipath_taps - 1
    length = n_echo * sps + 1
    taps = np.zeros(length, dtype=np.complex128)
    taps[0] = 1.0
    for j in range(1, n_echo + 1):
        decay = params.multipath_spread * (0.7 ** (j - 1))
        taps[j * sps] = decay * (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
    taps /= np.sqrt(np.sum(np.abs(taps) ** 2))  # unit energy: preserves power
    return taps


def apply_channel(
    clean: np.ndarray, params: ChannelParams, seed: int, sps: int = 1
) -> np.ndarray:
    """Apply multipath, CFO, IQ imbalance, and AWGN to a ``(channels, N)`` block.

    ``sps`` sets the symbol spacing of the multipath echoes.
    """
    clean = np.atleast_2d(np.asarray(clean, dtype=np.complex128))
    channels, n = clean.shape
    out = np.empty_like(clean)
    for c in range(channels):
        # The multipath realization is fixed by channel_seed if given, else drawn
        # from the episode seed; noise always uses the episode seed, and a
        # distinct stream so it never coincides with the tap draw.
        tap_seed = params.channel_seed if params.channel_seed is not None else (seed << 8) ^ c
        tap_rng = np.random.default_rng((int(tap_seed) << 8) ^ c)
        noise_rng = np.random.default_rng(((seed << 8) ^ c) ^ 0x9E3779B9)
        x = clean[c]
        # Multipath ISI (the thing to equalize).
        taps = _multipath_filter(tap_rng, params, sps)
        y = np.convolve(x, taps, mode="full")[:n]
        # Carrier frequency offset across the block.
        if params.cfo_cycles_per_block != 0.0:
            phase = 2 * np.pi * params.cfo_cycles_per_block * np.arange(n) / n
            y = y * np.exp(1j * phase)
        # IQ imbalance (gain + quadrature phase error).
        if params.iq_imbalance_db != 0.0 or params.iq_phase_deg != 0.0:
            g = 10.0 ** (params.iq_imbalance_db / 20.0)
            phi = np.deg2rad(params.iq_phase_deg)
            i = y.real * g
            q = y.imag * np.cos(phi) + y.real * np.sin(phi)
            y = i + 1j * q
        # AWGN at the requested SNR (signal normalized to unit power upstream).
        sig_power = float(np.mean(np.abs(y) ** 2))
        noise_power = sig_power / (10.0 ** (params.snr_db / 10.0))
        noise = np.sqrt(noise_power / 2.0) * (
            noise_rng.standard_normal(n) + 1j * noise_rng.standard_normal(n)
        )
        out[c] = y + noise
    return out
