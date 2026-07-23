"""Field deployment: host-side operation on real captures, with a safety gate.

The measured reality (see ``docs/DAY_ONE.md``) governs this module:

- Host-side operation on real captures is mechanically sound and safe (no
  bitstream, no flashing): decode IQ, run the operator, use the output.
- Fine-tuning is reliable **only against a known reference** (loopback of a
  transmitted calibration waveform, or a known lab emitter), where the clean
  truth is available and the coherence reward applies. Blind, over-the-air
  fine-tuning is *not* reliable -- it degraded true quality on most channels in
  testing -- so this module refuses to fine-tune without truth captures.
- Every fine-tune is validated on held-out truth captures before it is accepted;
  a candidate that does not strictly improve is rejected in favour of bypass, so
  a bad adaptation can never be shipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .capture import Capture
from .cma import train_operator
from .gym import Episode, EpisodeSpec
from .operator import NeuralOperator
from .reward import blind_quality, episode_reward
from .recovery import coherence
from .waveforms import WaveformProfile


# ---------------------------------------------------------------------------
# host-side execution on a capture
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldResult:
    output: np.ndarray                 # (channels, N) operator output
    blind_quality_in: float            # blind quality of the raw capture
    blind_quality_out: float           # blind quality of the operator output
    true_coherence_gain: Optional[float]  # only when the capture carries truth


def run_on_capture(operator, capture: Capture) -> FieldResult:
    """Run the operator host-side on one capture and report quality.

    ``operator`` is anything with ``forward(iq, fs)`` -- a float operator or an
    ``AbiExecutor`` (fixed-point, ABI-faithful). True coherence gain is reported
    only when the capture carries a reference (sim/twin/loopback).
    """
    out = operator.forward(capture.iq, capture.fs_hz)
    sps = capture.sps or 4
    qi = float(np.mean([blind_quality(capture.iq[c], sps)[0] for c in range(capture.channels)]))
    qo = float(np.mean([blind_quality(out[c], sps)[0] for c in range(capture.channels)]))
    gain = None
    if capture.clean is not None:
        gain = float(np.mean([
            coherence(out[c], capture.clean[c]) - coherence(capture.iq[c], capture.clean[c])
            for c in range(capture.channels)
        ]))
    return FieldResult(output=out, blind_quality_in=qi, blind_quality_out=qo,
                       true_coherence_gain=gain)


# ---------------------------------------------------------------------------
# capture replay as a gym, for known-signal fine-tuning
# ---------------------------------------------------------------------------
class CaptureReplayGym:
    """Presents a fixed set of captures as training episodes (no synthesis).

    Used to fine-tune against real, recorded captures. Requires each capture to
    carry ``clean`` (a known reference); the coherence reward is undefined
    otherwise, which is the point -- blind fine-tuning is intentionally not
    supported here.
    """

    def __init__(self, captures: List[Capture]) -> None:
        if any(c.clean is None for c in captures):
            raise ValueError("known-signal fine-tuning requires captures with a reference")
        self.captures = list(captures)
        self.n_channels = captures[0].channels

    def sample_spec(self, rng, n_samples: int = 4096, noise_prob: float = 0.0) -> EpisodeSpec:
        idx = int(rng.integers(0, len(self.captures)))
        cap = self.captures[idx]
        from .channel import ChannelParams
        return EpisodeSpec(
            profile=WaveformProfile("qpsk", sps=cap.sps or 4),
            channel=ChannelParams(), fs_hz=cap.fs_hz,
            n_samples=min(n_samples, cap.iq.shape[1]), seed=idx,
            n_channels=cap.channels, is_noise=False,
        )

    def realize(self, spec: EpisodeSpec) -> Episode:
        cap = self.captures[spec.seed]
        n = spec.n_samples
        return Episode(spec=spec, observed=cap.iq[:, :n], clean=cap.clean[:, :n], synth=None)


# ---------------------------------------------------------------------------
# known-signal fine-tune, with a mandatory validation gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FineTuneOutcome:
    accepted: bool
    validation_gain: float            # mean true coherence gain on held-out captures
    operator: NeuralOperator          # the accepted operator, or the warm-start fallback


def finetune_known_signal(
    template: NeuralOperator,
    train_captures: List[Capture],
    validation_captures: List[Capture],
    generations: int = 16,
    batch: int = 6,
    n_samples: int = 1024,
    sigma0: float = 0.3,
    seed: int = 0,
    accept_threshold: float = 0.01,
) -> FineTuneOutcome:
    """Fine-tune against known-reference captures and gate on held-out truth.

    Trains on the coherence (truth) reward, then validates on captures the
    training never saw. The result is accepted only if the mean true coherence
    gain clears ``accept_threshold``; otherwise the warm-start (bypass-equivalent)
    is returned, so a fine-tune that does not help is never shipped.
    """
    if any(c.clean is None for c in validation_captures):
        raise ValueError("validation requires captures with a reference")
    gym = CaptureReplayGym(train_captures)
    history = train_operator(template, gym, episode_reward, generations=generations,
                             batch=batch, n_samples=n_samples, sigma0=sigma0, seed=seed)
    candidate = template.with_adapted_vector(history.best_vector)

    gains = []
    for cap in validation_captures:
        n = min(n_samples, cap.iq.shape[1])
        out = candidate.forward(cap.iq[:, :n], cap.fs_hz)
        gains.append(float(np.mean([
            coherence(out[c], cap.clean[c, :n]) - coherence(cap.iq[c, :n], cap.clean[c, :n])
            for c in range(cap.channels)
        ])))
    validation_gain = float(np.mean(gains))
    if validation_gain >= accept_threshold:
        return FineTuneOutcome(True, validation_gain, candidate)
    return FineTuneOutcome(False, validation_gain, template)
