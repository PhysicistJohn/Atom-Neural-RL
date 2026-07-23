"""The training environment: episodes drawn from a profile catalog and channel.

An episode bundles a clean emitter, an impaired observation, and the ground
truth, all reproducible from a seed. Two invariants enforced here because the
verification flagged them:

- **Rate is drawn conditioned on occupied bandwidth.** Occupied bandwidth is a
  fixed fraction ``(1+beta)/sps < 1`` of the sample rate by construction
  (``sps >= 2``), so ``bandwidth_hz < fs`` always holds and no episode violates
  Nyquist. The physical sample rate is drawn across ``[F0/8, F0]`` so the
  master-rate-anchored kernel is trained under the full range of transport it
  will see in deployment.
- **Channels are a tensor dimension.** ``n_channels`` streams are produced as
  ``(channels, N)``; each channel shares the emitter but draws its own
  impairment realization.

Signal-free probe episodes (``is_noise``) exist for the honesty gate: an
operator must not manufacture apparent signal quality from pure noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .channel import ChannelParams, apply_channel
from .waveforms import MODULATIONS, SynthResult, WaveformProfile, synthesize
from .zplane import F0_HZ

FS_MIN_HZ = F0_HZ / 8.0


def default_catalog() -> List[WaveformProfile]:
    """A canonical profile catalog: every modulation over a grid of sps/rolloff."""
    profiles: List[WaveformProfile] = []
    for mod in MODULATIONS:
        for sps in (2, 4, 8):
            for rolloff in (0.2, 0.35, 0.5):
                profiles.append(WaveformProfile(mod, sps=sps, rolloff=rolloff))
    return profiles


@dataclass(frozen=True)
class EpisodeSpec:
    """A fully-specified, reproducible episode."""

    profile: WaveformProfile
    channel: ChannelParams
    fs_hz: float
    n_samples: int
    seed: int
    n_channels: int = 1
    is_noise: bool = False

    def occupied_bandwidth_hz(self) -> float:
        return self.profile.occupied_bandwidth_fraction() * self.fs_hz


@dataclass(frozen=True)
class Episode:
    """A realized episode: impaired input, clean truth, and symbols."""

    spec: EpisodeSpec
    observed: np.ndarray  # (channels, N) impaired
    clean: np.ndarray     # (channels, N) noiseless, channel-free reference
    synth: Optional[SynthResult]  # None for noise probes


class Gym:
    """Draws and realizes episodes from a catalog with domain randomization."""

    def __init__(
        self,
        catalog: Optional[Sequence[WaveformProfile]] = None,
        n_channels: int = 1,
    ) -> None:
        self.catalog = list(catalog) if catalog is not None else default_catalog()
        self.n_channels = n_channels

    # -- specification ----------------------------------------------------
    def sample_spec(
        self, rng: np.random.Generator, n_samples: int = 4096, noise_prob: float = 0.0
    ) -> EpisodeSpec:
        is_noise = bool(rng.random() < noise_prob)
        profile = self.catalog[int(rng.integers(0, len(self.catalog)))]
        channel = ChannelParams(
            snr_db=float(rng.uniform(5.0, 30.0)),
            multipath_taps=int(rng.integers(1, 5)),
            multipath_spread=float(rng.uniform(0.1, 0.6)),
            cfo_cycles_per_block=float(rng.uniform(-2.0, 2.0)),
            iq_imbalance_db=float(rng.uniform(-0.5, 0.5)),
            iq_phase_deg=float(rng.uniform(-3.0, 3.0)),
        )
        fs = float(np.exp(rng.uniform(np.log(FS_MIN_HZ), np.log(F0_HZ))))
        seed = int(rng.integers(0, 2 ** 31 - 1))
        return EpisodeSpec(
            profile=profile,
            channel=channel,
            fs_hz=fs,
            n_samples=n_samples,
            seed=seed,
            n_channels=self.n_channels,
            is_noise=is_noise,
        )

    # -- realization ------------------------------------------------------
    def realize(self, spec: EpisodeSpec) -> Episode:
        n = spec.n_samples
        if spec.is_noise:
            clean = np.zeros((spec.n_channels, n), dtype=np.complex128)
            rng = np.random.default_rng(spec.seed)
            observed = (
                rng.standard_normal((spec.n_channels, n))
                + 1j * rng.standard_normal((spec.n_channels, n))
            ) / np.sqrt(2)
            return Episode(spec=spec, observed=observed, clean=clean, synth=None)
        synth = synthesize(spec.profile, n, spec.seed)
        clean = np.tile(synth.iq, (spec.n_channels, 1))
        observed = apply_channel(clean, spec.channel, spec.seed, spec.profile.sps)
        return Episode(spec=spec, observed=observed, clean=clean, synth=synth)

    # -- train / held-out split ------------------------------------------
    def leave_one_modulation_out(self, held_out: str) -> Tuple["Gym", "Gym"]:
        """Split into (train, held-out) gyms by modulation, for G2 non-inferiority."""
        train = [p for p in self.catalog if p.modulation != held_out]
        heldout = [p for p in self.catalog if p.modulation == held_out]
        if not heldout:
            raise ValueError(f"no profiles with modulation {held_out!r}")
        return Gym(train, self.n_channels), Gym(heldout, self.n_channels)
