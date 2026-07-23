"""Host-side reward stack, v1 = signal quality. Paired, differential, anchored.

Every reward is a *differential* between the operator output and bypass on the
**same** episode buffer, so episode difficulty cancels and the baseline is never
recomputed against a different draw. The composition, with the verification's
corrections baked in:

- **Fidelity anchor (dominant).** Truth-referenced alignment error to the clean
  waveform. Immune to gain inflation (scale fitted out) and to content collapse
  (discarding signal increases the error). This term carries the reward.
- **Blind ISI differential (secondary, lock-gated).** Counts only where blind
  recovery reports a lock (converged + residual ISI under threshold), so an
  unconverged proxy cannot contribute noise.
- **Blind SNR differential (clamped).** Contribution clipped to +/-1 so the most
  hackable metric cannot dominate.
- **Full-band power-conservation penalty.** Punishes gross power change relative
  to the input, closing the residual gain/noise-floor-sculpting routes.

Non-finite operator output scores a large negative reward, so CMA-ES treats a
numerically diverged operator as strongly bad rather than being NaN-poisoned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .recovery import alignment_error, blind_recover
from .gym import Episode

_DIVERGED_REWARD = -10.0


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the v1 signal-quality reward. Anchor-dominant by design."""

    w_anchor: float = 1.0
    w_isi: float = 0.3
    w_snr: float = 0.1
    snr_clip: float = 1.0
    w_power: float = 0.5
    power_deadzone: float = 0.405  # ln(1.5): modest power change is unpenalized
    isi_lock_threshold: float = 0.22


def _finite(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(x.view(np.float64))))


def stream_reward(
    operator_out: np.ndarray,
    bypass_out: np.ndarray,
    clean: np.ndarray,
    observed_in: np.ndarray,
    sps: int,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Differential signal-quality reward for one stream (operator vs bypass)."""
    operator_out = np.asarray(operator_out, dtype=np.complex128)
    if not _finite(operator_out):
        return _DIVERGED_REWARD

    # Fidelity anchor: lower alignment error is better, so the improvement is
    # bypass_error - operator_error.
    anchor = alignment_error(bypass_out, clean) - alignment_error(operator_out, clean)

    # Blind ISI and SNR differentials, gated on the BYPASS (input) lock -- never
    # the operator's own output. The operator cannot influence whether its input
    # was recoverable, so it cannot open this credit channel by shaping its
    # output (e.g. turning noise into a constant-modulus signal). On a signal-free
    # probe the input never locks, so neither term can pay out, and the hack is
    # closed at the source rather than left for the honesty gate to reject.
    rec_by = blind_recover(bypass_out, sps)
    if rec_by.locked(config.isi_lock_threshold):
        rec_op = blind_recover(operator_out, sps)
        isi = rec_by.residual_isi - rec_op.residual_isi
        snr = float(np.clip(rec_op.snr_db - rec_by.snr_db, -config.snr_clip, config.snr_clip))
    else:
        isi = 0.0
        snr = 0.0

    # Full-band power conservation penalty relative to the input, with a deadzone
    # so that legitimate equalization (which changes power modestly) is free while
    # gross inflation or collapse is punished.
    p_op = float(np.mean(np.abs(operator_out) ** 2))
    p_in = float(np.mean(np.abs(observed_in) ** 2))
    log_ratio = abs(np.log((p_op + 1e-12) / (p_in + 1e-12)))
    power_penalty = max(0.0, log_ratio - config.power_deadzone)

    return (
        config.w_anchor * anchor
        + config.w_isi * isi
        + config.w_snr * snr
        - config.w_power * power_penalty
    )


def episode_reward(operator, episode: Episode, config: RewardConfig = RewardConfig()) -> float:
    """Mean paired-differential reward over an episode's channels.

    ``operator`` is any object with ``forward(iq, fs_hz) -> (channels, N)``.
    Bypass is the observed input (the identity operator's output).
    """
    observed = episode.observed
    fs = episode.spec.fs_hz
    op_out = operator.forward(observed, fs)
    if not _finite(op_out):
        return _DIVERGED_REWARD
    sps = episode.spec.profile.sps
    rewards = [
        stream_reward(op_out[c], observed[c], episode.clean[c], observed[c], sps, config)
        for c in range(observed.shape[0])
    ]
    return float(np.mean(rewards))


def probe_reward(operator, noise_episode: Episode, config: RewardConfig = RewardConfig()) -> float:
    """Reward on a signal-free episode. An honest operator scores ~0; the honesty
    gate rejects any bank whose magnitude here exceeds a small tolerance."""
    return episode_reward(operator, noise_episode, config)
