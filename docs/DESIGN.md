# Atom-Neural-RL design

This repository implements P1 of the neural-operator initiative: the entire
zero-hardware core of a z-plane invariant complex Fourier neural operator learned
under reinforcement learning. It is `numpy` + stdlib `unittest` only, matching the
suite's zero-dependency culture; the source gate promotes `RuntimeWarning`s to
errors.

## The invariance contract (`zplane.py`, `tests/test_invariance.py`)

The learned object is a single rational function on the z-plane in product form,
anchored at the master rate `F0 = 61.44 MSPS`.

- **Discretization (N) invariance is exact for the response.** `H(e^{jw})` is a
  closed-form function of normalized frequency; the same physical frequency reads
  identically at every FFT size (`2^10 .. 2^16`).
- **Applying by N-point circular convolution has a certified error**, the honest
  bound `eps(N) <= 2 R rho^N / (1 - rho)` with `R = sum|r_j|` a *certified*
  residue norm and `rho` the pole radius. The naive `rho^N <= eps` shortcut drops
  the `1/(1-rho)` prefactor (~124x at `N=2^10`, ~40 dB of overclaim) and is
  rejected in `max_stable_radius_for`.
- **Rate handling is master-rate anchoring, no fallback.** Deploying at
  `fs <= F0` maps `p -> p^(F0/fs)`; the exponent is always `>= 1`, so poles only
  move inward, stability and the certificate improve automatically, and the
  radial projection can never fire in deployment. Invariance on the z-plane is
  covariance in Hz -- the physical frequency of a feature is preserved by
  transport. The one approximation named honestly: the per-sample nonlinearity
  generates harmonic content that aliases differently across rates, so the *full
  nonlinear* operator's rate invariance is certified-approximate; the linear path
  is exact.

## The operator (`operator.py`)

Complex FNO: lift `(W, C_in)`, `L` layers of diagonal spectral kernels plus a
pointwise `W x W` mixing matrix and `modReLU` (`b <= 0`), project `(C_out, W)`.
No additive biases (they would break the global-phase equivariance every reward
shares). Channels are a tensor dimension: `(channels, N)` throughout, any count.
Configs `C0..C4`; on hardware, RL adapts only the spectral kernels (`~10^2`
reals), which is what makes CMA-ES affordable.

Two initialization facts matter and are enforced:

- The **identity** operator reproduces its input exactly (safe/rollback bank,
  twin bit-exactness vector).
- Training starts from **`warm_start`** (poles and zeros co-located, so `H == 1`
  exactly but in the responsive interior of the search space). Starting from the
  identity packing lands the search on a flat plateau because the identity
  saturates the pole-radius sigmoid.

## Fixed point (`fixedpoint.py`)

The operator-added datapath the fabric realizes: Q1.15 shared-exponent weight
tables, 24-bit block-float spectral data, exponent-compensated `modReLU`. The
shared FFT core is pre-existing fabric and out of scope. Weight quantization is
~-90 dBc.

## The gym (`waveforms.py`, `channel.py`, `gym.py`)

Deterministic RRC PSK/QAM synthesis, seeded symbol-spaced multipath + AWGN + CFO
+ IQ imbalance, episodes drawn from a profile catalog with the sample rate drawn
across `[F0/8, F0]` and always Nyquist-safe by construction. `channel_seed` fixes
a channel for the fine-tune regime; leave-one-modulation-out builds the G2 split.

## Reward (`recovery.py`, `reward.py`)

One core truth, not a defended sum of proxies. Signal quality is **coherence to
the clean transmitted waveform** (`recovery.coherence`): the fraction of the
operator output that is a genuine copy of the true signal after the
coherent-receiver nuisance group -- gain, carrier phase, timing, carrier
frequency -- is fitted out. What remains is genuine quality: ISI, noise,
distortion. The reward is the improvement in that one quantity, operator vs
bypass on the same buffer.

Because coherence is gain-invariant (`gamma^2 = SNR/(1+SNR)` under noise, so a
uniform scale is invisible), collapse-punishing (an orthogonal output has
coherence 0), and self-regularizing (added out-of-band energy raises the norm,
not the correlation, so coherence falls), every hacking route is closed by the
definition. There is no power penalty, deadzone, SNR clip, lock-gate, or weight
to tune -- all of which existed only to defend gameable proxies. `test_reward.py`
proves each hack fails *by construction*, and that a pure gain earns exactly zero
without any term to enforce it.

The blind CMA-recovery metrics are not part of this reward. They are the
*hardware* reward (`blind_episode_reward`), for the regime where the clean truth
is unavailable, credited only when the input itself was recoverable (a
definitional gate, not a patch). Before they are trusted, `proxy_validity`
certifies in sim that the blind reward ranks operators the same way coherence
does (correlation ~0.55 across the operator population). Signal-free episodes
carry no waveform, so the truth reward simply excludes them; honesty on noise is
a property of the blind path, whose recoverability gate makes fabricated
improvement impossible.

## Optimizer and gates (`cma.py`, `gates.py`)

CMA-ES (mirrored sampling, common random numbers) over the small adapted vector;
the parameter map guarantees stable poles and `b <= 0` for any real vector, so
the search is unconstrained yet always legal. Gates G1 (strict coherence
improvement with bootstrap CI + effect-size floor), G2 (leave-one-modulation-out
non-inferiority), G3 (single-violation honesty probes on the blind path), G3b
(proxy validity: the blind reward tracks coherence across the operator space),
G4 (quantized realizability). G5 (twin acceptance) lives in the twin repo.
`tests/test_cma.py` runs a real training loop that must actually learn to
equalize a planted channel (fixed-rate fine-tune reaches ~0.14 coherence gain).

## Banks (`bank.py`)

Content-addressed (sha256) serialization with a CRC32, a certified LOG2_N
validity range in the header, and lineage (parent, seed, reward/env/eval
digests). `emit_tables(N)` produces the dense per-mode quantized tables the fabric
loads, and refuses sizes outside the certified range.

## What is deliberately out of scope here

P2 firmware interface JSON v2 and guest, P3 twin device model and G5, P4 RTL, P5
board. The optional JAX gradient-pretraining backend (`[pretrain]` extra) is a
later drop-in; CMA-ES is gradient-free and is the normative optimizer, so its
absence never blocks the suite.
