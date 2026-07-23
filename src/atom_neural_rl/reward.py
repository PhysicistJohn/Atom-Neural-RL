"""The reward, built on one core truth instead of a defended sum of proxies.

Signal quality is **coherence to the clean transmitted waveform** (``recovery.
coherence``): the fraction of the operator output that is a genuine copy of the
true signal, after the coherent-receiver nuisance group (gain, phase, timing,
carrier frequency) is fitted out. The reward is the improvement in that one
quantity, operator versus bypass, on the same buffer:

    reward = coherence(operator_out, clean) - coherence(bypass, clean)

Because coherence is gain/phase invariant, collapse-punishing, and
self-regularizing, every failure mode is closed by the definition rather than by
a patch. There is no power penalty, no deadzone, no SNR clip, no lock-gate, and
no weight to tune. A gain-inflated output has identical coherence (reward 0); a
collapsed output has coherence 0 (reward < 0); an output with added out-of-band
energy has lower coherence (reward < 0).

Signal-free episodes carry no waveform to be faithful to, so they are simply not
part of this reward (it returns 0). Honesty on noise is a property of the
*blind* path below, which is the only path that runs where truth is absent.

The blind path (:func:`blind_episode_reward`) is the hardware reward: it uses the
CMA recovery proxy because on a real capture there is no clean truth. It is
credited only when the input itself was recoverable (the bypass locks), a gate
that is definitional -- you cannot improve the recovery of a signal that was not
there -- not a patch. Before it is ever trusted, :func:`proxy_validity` certifies
in sim that it tracks the coherence reward.
"""
from __future__ import annotations

import numpy as np

from .gym import Episode
from .recovery import blind_recover, coherence

_DIVERGED_REWARD = -10.0


def _finite(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(x.view(np.float64))))


# ---------------------------------------------------------------------------
# the reward: improvement in coherence to the clean waveform (sim / twin)
# ---------------------------------------------------------------------------
def episode_reward(operator, episode: Episode) -> float:
    """Improvement in coherence to the clean waveform, averaged over channels.

    Returns 0 for a signal-free episode (no waveform to be faithful to) and a
    large negative reward for a numerically diverged operator.
    """
    if episode.spec.is_noise:
        return 0.0
    observed = episode.observed
    op_out = operator.forward(observed, episode.spec.fs_hz)
    if not _finite(op_out):
        return _DIVERGED_REWARD
    gains = [
        coherence(op_out[c], episode.clean[c]) - coherence(observed[c], episode.clean[c])
        for c in range(observed.shape[0])
    ]
    return float(np.mean(gains))


# ---------------------------------------------------------------------------
# the blind proxy reward (hardware regime, no truth available)
# ---------------------------------------------------------------------------
def blind_quality(stream: np.ndarray, sps: int, isi_lock_threshold: float = 0.22):
    """Blind recovery quality of one stream, plus its lock state.

    Quality is ``1 / (1 + residual_isi)`` in (0, 1]; higher is better. Returned
    with the lock flag so the caller can apply the recoverability gate.
    """
    rec = blind_recover(stream, sps)
    return 1.0 / (1.0 + rec.residual_isi), rec.locked(isi_lock_threshold)


def blind_episode_reward(operator, episode: Episode) -> float:
    """Hardware-regime reward: improvement in blind recovery quality.

    Credited per channel only when the *input* (bypass) was recoverable -- a
    definitional gate the operator cannot open by shaping its own output, so a
    signal-free capture (whose input never locks) can never pay out.
    """
    observed = episode.observed
    op_out = operator.forward(observed, episode.spec.fs_hz)
    if not _finite(op_out):
        return _DIVERGED_REWARD
    sps = episode.spec.profile.sps
    gains = []
    for c in range(observed.shape[0]):
        q_by, by_locked = blind_quality(observed[c], sps)
        if not by_locked:
            gains.append(0.0)
            continue
        q_op, _ = blind_quality(op_out[c], sps)
        gains.append(q_op - q_by)
    return float(np.mean(gains)) if gains else 0.0


# ---------------------------------------------------------------------------
# the certificate binding the proxy to the truth
# ---------------------------------------------------------------------------
def _proxy_correlation_once(operator, gym, n_operators, count, seed, n_samples) -> float:
    from .operator import NeuralOperator  # local import avoids a cycle at load

    rng = np.random.default_rng(seed)
    episodes = [gym.realize(gym.sample_spec(rng, n_samples=n_samples)) for _ in range(count)]
    base = operator.adapted_vector()
    warm = NeuralOperator.warm_start(operator.config).adapted_vector()
    truth, blind = [], []
    for k in range(n_operators):
        alpha = 1.6 * k / max(n_operators - 1, 1)
        vec = warm + alpha * (base - warm)
        op = operator.with_adapted_vector(vec)
        truth.append(float(np.mean([episode_reward(op, ep) for ep in episodes])))
        blind.append(float(np.mean([blind_episode_reward(op, ep) for ep in episodes])))
    truth = np.asarray(truth)
    blind = np.asarray(blind)
    if np.std(truth) < 1e-9 or np.std(blind) < 1e-9:
        return 0.0
    tr = np.argsort(np.argsort(truth)).astype(float)
    br = np.argsort(np.argsort(blind)).astype(float)
    return float(np.corrcoef(tr, br)[0, 1])


def proxy_validity(
    operator,
    gym,
    n_operators: int = 9,
    count: int = 6,
    seed: int = 0,
    n_samples: int = 1024,
    repeats: int = 4,
) -> float:
    """Rank correlation between the truth reward and the blind proxy along a
    quality ladder of operators -- the certificate that the blind (hardware)
    reward ranks operators the same way the coherence (truth) reward does.

    The ladder is a deterministic interpolation from the do-nothing warm start
    (alpha 0) through ``operator`` (alpha 1) to an overshoot (alpha ~1.6), so it
    spans genuinely good to genuinely degraded along the *meaningful* axis the
    operator was trained on -- unlike random perturbations of a neutral operator,
    which only span neutral-to-bad and give a noisy near-zero correlation.

    The blind CMA-dispersion metric is a moderate, noisy proxy, so a single
    correlation estimate has high variance; the reported value is averaged over
    ``repeats`` independent episode draws. A high value means optimizing the blind
    reward on hardware optimizes true signal quality, so the proxy is trustworthy.
    This one measured guarantee replaces the per-term anti-hacking patches; the
    residual safety on real hardware is periodic re-validation against truth in
    sim, which the metric's honest, moderate strength makes explicit rather than
    hiding behind a defended composite.
    """
    vals = [
        _proxy_correlation_once(operator, gym, n_operators, count, seed + 100 * r, n_samples)
        for r in range(repeats)
    ]
    return float(np.mean(vals))
