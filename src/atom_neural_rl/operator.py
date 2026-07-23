"""The complex Fourier neural operator, float reference.

Structure (one layer):

    v_{l+1}[n] = modReLU_b( M_l @ v_l[n] + IFFT[ D_l(omega) .* FFT[v_l] ][n] )

with ``M_l`` a complex ``W x W`` pointwise (1x1) mixing matrix carrying all
cross-channel coupling, ``D_l`` a *diagonal* bank of ``W`` scalar rational
kernels (the spectral path), and ``modReLU`` the complex soft-threshold. There
are **no additive biases**: any nonzero additive bias makes a layer affine and
breaks the global-phase equivariance that every host-side reward shares, so
biases are zero by construction and ``b`` (the modReLU magnitude offset,
``b <= 0``) is the only offset in the network.

Lift ``P`` (``W x 2``) and project ``Q`` (``2 x W``) are frozen from sim
pretraining; on hardware, RL adapts only the spectral kernels and thresholds --
order 10^2 real parameters -- which is what makes CMA-ES episodes affordable.

The identity configuration (``P = I``, ``M = 0``, kernels identity, ``b = 0``)
reproduces the input exactly and is the named safe/rollback bank and the twin
bit-exactness test vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .zplane import RHO_MAX, RationalKernel


# ---------------------------------------------------------------------------
# nonlinearity
# ---------------------------------------------------------------------------
def modrelu(z: np.ndarray, b: float) -> np.ndarray:
    """Complex soft-threshold ``max(0, |z| + b) * z/|z|`` with ``b <= 0``.

    Phase-equivariant, 1-Lipschitz, and the identity at ``b = 0``. The zero-
    magnitude samples map to zero (the limit is well defined).
    """
    mag = np.abs(z)
    scale = np.maximum(0.0, mag + b)
    out = np.zeros_like(z)
    nz = mag > 0
    out[nz] = scale[nz] * (z[nz] / mag[nz])
    return out


# ---------------------------------------------------------------------------
# configuration + frozen backbone
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperatorConfig:
    """Shape of an operator. ``sections`` is the pole/zero order ``K`` per kernel."""

    width: int
    layers: int
    sections: int
    in_channels: int = 2
    out_channels: int = 2
    name: str = "custom"

    @property
    def kernel_count(self) -> int:
        return self.width * self.layers

    @staticmethod
    def diagonal_for_channels(
        channels: int, sections: int = 16, name: str = "diag"
    ) -> "OperatorConfig":
        """A per-channel diagonal operator over an arbitrary channel count.

        Channels are a tensor dimension throughout: the operator, the gym, the
        channel model, and the reward stack all carry data as ``(channels, N)``
        with ``channels`` free. This convenience builds a width == channels,
        single-layer config so each channel gets its own rational kernel with no
        forced cross-channel mixing (the frozen mix is zero for the identity
        backbone; cross-antenna coupling is available by loading a non-zero mix).
        """
        return OperatorConfig(
            width=channels,
            layers=1,
            sections=sections,
            in_channels=channels,
            out_channels=channels,
            name=name,
        )

    @property
    def adapted_dim(self) -> int:
        """Real-parameter count of the RL-adapted vector (kernels + thresholds).

        Per kernel: ``2K`` pole reals (radius,angle) + ``2K`` zero reals
        + ``2`` gain reals = ``4K + 2``. Plus one threshold real per kernel.
        """
        per_kernel = 4 * self.sections + 2 + 1
        return self.kernel_count * per_kernel


# Named configurations from the design spec (approximate the published budgets;
# the exact counts follow from K and are reported by ``adapted_dim``).
C0 = OperatorConfig(width=1, layers=1, sections=16, in_channels=1, out_channels=1, name="C0")
C1 = OperatorConfig(width=2, layers=1, sections=16, name="C1")
C2 = OperatorConfig(width=4, layers=2, sections=8, name="C2")
C3 = OperatorConfig(width=4, layers=4, sections=8, name="C3")
C4 = OperatorConfig(width=32, layers=4, sections=8, name="C4")

NAMED_CONFIGS = {c.name: c for c in (C0, C1, C2, C3, C4)}


@dataclass(frozen=True)
class FrozenBackbone:
    """The pretrained, RL-frozen linear scaffold: lift, per-layer mixes, project."""

    lift: np.ndarray            # (W, in_channels) complex
    mixes: Sequence[np.ndarray]  # L of (W, W) complex
    project: np.ndarray         # (out_channels, W) complex

    @staticmethod
    def identity(config: OperatorConfig) -> "FrozenBackbone":
        """Backbone that, with identity kernels and ``b = 0``, is a pass-through.

        Requires ``width == in_channels == out_channels`` (the C0/C1 shapes);
        for wider configs it produces a lift that embeds the input in the first
        ``in_channels`` hidden lanes and a project that reads them back.
        """
        w, ci, co = config.width, config.in_channels, config.out_channels
        lift = np.zeros((w, ci), dtype=np.complex128)
        for i in range(min(w, ci)):
            lift[i, i] = 1.0
        project = np.zeros((co, w), dtype=np.complex128)
        for i in range(min(co, w)):
            project[i, i] = 1.0
        mixes = [np.zeros((w, w), dtype=np.complex128) for _ in range(config.layers)]
        return FrozenBackbone(lift, mixes, project)


# ---------------------------------------------------------------------------
# the operator
# ---------------------------------------------------------------------------
class NeuralOperator:
    """A configured complex FNO with adaptable spectral kernels."""

    def __init__(
        self,
        config: OperatorConfig,
        backbone: FrozenBackbone,
        kernels: List[List[RationalKernel]],
        thresholds: np.ndarray,
    ) -> None:
        if len(kernels) != config.layers or any(len(row) != config.width for row in kernels):
            raise ValueError("kernels must be shaped (layers, width)")
        if thresholds.shape != (config.layers, config.width):
            raise ValueError("thresholds must be shaped (layers, width)")
        if np.any(thresholds > 0.0):
            raise ValueError("all modReLU thresholds must be <= 0")
        self.config = config
        self.backbone = backbone
        self.kernels = kernels
        self.thresholds = np.asarray(thresholds, dtype=np.float64)

    # -- construction -----------------------------------------------------
    @classmethod
    def identity(cls, config: OperatorConfig) -> "NeuralOperator":
        backbone = FrozenBackbone.identity(config)
        kernels = [
            [RationalKernel.identity() for _ in range(config.width)]
            for _ in range(config.layers)
        ]
        thresholds = np.zeros((config.layers, config.width), dtype=np.float64)
        return cls(config, backbone, kernels, thresholds)

    @classmethod
    def warm_start(cls, config: OperatorConfig, radius: float = 0.5) -> "NeuralOperator":
        """A responsive near-identity operator: ``H(z) == 1`` but parameterized in
        the responsive interior of the search space (not the saturated tail).

        Each kernel places ``K`` poles and ``K`` zeros at the *same* locations
        (radius ``radius``, angles spread around the circle), so the pole/zero
        product cancels to unit response exactly, while the packed logits sit
        near zero where CMA-ES perturbations actually move the filter. Starting a
        search from :meth:`identity` instead lands on a flat reward plateau
        because the identity packing saturates the radius sigmoid.
        """
        backbone = FrozenBackbone.identity(config)
        k = config.sections
        angles = 2.0 * np.pi * np.arange(k) / max(k, 1)
        loc = radius * np.exp(1j * angles)
        kernels = [
            [RationalKernel(1.0 + 0j, loc.copy(), loc.copy()) for _ in range(config.width)]
            for _ in range(config.layers)
        ]
        thresholds = np.zeros((config.layers, config.width), dtype=np.float64)
        return cls(config, backbone, kernels, thresholds)

    # -- forward ----------------------------------------------------------
    def forward(self, iq: np.ndarray, fs_hz: float) -> np.ndarray:
        """Apply the operator to an IQ block ``(in_channels, N)`` at rate ``fs_hz``.

        Returns ``(out_channels, N)`` complex. Kernels are transported from the
        master anchor to ``fs_hz`` before evaluation, so this is the identical
        pipeline used in training, twin, and hardware.
        """
        iq = np.asarray(iq, dtype=np.complex128)
        if iq.ndim != 2 or iq.shape[0] != self.config.in_channels:
            raise ValueError(f"iq must be ({self.config.in_channels}, N)")
        n = iq.shape[1]
        # einsum (not ``@``) for the channel-mixing contractions: numpy 2.0's
        # complex matmul kernel emits spurious FP-flag RuntimeWarnings that would
        # flood CMA-ES logs and mask genuine overflow. einsum is warning-clean.
        v = np.einsum("wc,cn->wn", self.backbone.lift, iq)  # (W, N)
        for layer in range(self.config.layers):
            spectral = np.empty_like(v)
            for w in range(self.config.width):
                kernel = self.kernels[layer][w]
                if fs_hz != kernel.anchor_hz:
                    kernel = kernel.transport(fs_hz)
                resp = kernel.response_on_grid(n)
                spectral[w] = np.fft.ifft(resp * np.fft.fft(v[w]))
            mixed = np.einsum("ij,jn->in", self.backbone.mixes[layer], v)  # (W, N)
            pre = mixed + spectral
            for w in range(self.config.width):
                v[w] = modrelu(pre[w], float(self.thresholds[layer, w]))
        return np.einsum("cw,wn->cn", self.backbone.project, v)  # (out_channels, N)

    @classmethod
    def build(cls, config: OperatorConfig, backbone: FrozenBackbone, vector: np.ndarray) -> "NeuralOperator":
        """Reconstruct an operator from a config, backbone, and adapted vector."""
        placeholder = [
            [RationalKernel.identity() for _ in range(config.width)]
            for _ in range(config.layers)
        ]
        base = cls(config, backbone, placeholder, np.zeros((config.layers, config.width)))
        return base.with_adapted_vector(vector)

    # -- adapted parameter vector (for CMA-ES) ----------------------------
    def adapted_vector(self) -> np.ndarray:
        """Pack kernels + thresholds into the unconstrained real vector CMA sees."""
        parts: List[np.ndarray] = []
        for layer in range(self.config.layers):
            for w in range(self.config.width):
                parts.append(_kernel_to_unconstrained(self.kernels[layer][w], self.config.sections))
                parts.append(np.array([_threshold_to_unconstrained(self.thresholds[layer, w])]))
        return np.concatenate(parts)

    def with_adapted_vector(self, vector: np.ndarray) -> "NeuralOperator":
        """Return a new operator with kernels/thresholds set from ``vector``.

        The mapping guarantees stable poles (``|p| <= RHO_MAX``) and ``b <= 0``
        for *any* real vector, so CMA-ES searches an unconstrained space and can
        never propose an illegal operator.
        """
        k = self.config.sections
        per_kernel = 4 * k + 2 + 1
        expected = self.config.kernel_count * per_kernel
        if vector.shape != (expected,):
            raise ValueError(f"expected adapted vector of length {expected}")
        kernels: List[List[RationalKernel]] = []
        thresholds = np.zeros((self.config.layers, self.config.width), dtype=np.float64)
        cursor = 0
        for layer in range(self.config.layers):
            row: List[RationalKernel] = []
            for w in range(self.config.width):
                chunk = vector[cursor : cursor + 4 * k + 2]
                row.append(_unconstrained_to_kernel(chunk, k))
                cursor += 4 * k + 2
                thresholds[layer, w] = _unconstrained_to_threshold(vector[cursor])
                cursor += 1
            kernels.append(row)
        return NeuralOperator(self.config, self.backbone, kernels, thresholds)


# ---------------------------------------------------------------------------
# constrained <-> unconstrained parameter maps
# ---------------------------------------------------------------------------
# Zeros are allowed outside the unit disk (non-minimum-phase); poles are capped.
_ZERO_SCALE = 2.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(y: float, scale: float) -> float:
    r = min(max(y / scale, 1e-6), 1.0 - 1e-6)
    return float(np.log(r / (1.0 - r)))


def _softplus(x: float) -> float:
    return float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))


def _unconstrained_to_kernel(chunk: np.ndarray, k: int) -> RationalKernel:
    pole_r = RHO_MAX * _sigmoid(chunk[0:k])
    pole_a = chunk[k : 2 * k]
    zero_r = _ZERO_SCALE * _sigmoid(chunk[2 * k : 3 * k])
    zero_a = chunk[3 * k : 4 * k]
    gain = complex(chunk[4 * k], chunk[4 * k + 1])
    poles = pole_r * np.exp(1j * pole_a)
    zeros = zero_r * np.exp(1j * zero_a)
    return RationalKernel(gain, poles, zeros)


def _kernel_to_unconstrained(kernel: RationalKernel, k: int) -> np.ndarray:
    poles = kernel.poles
    zeros = kernel.zeros
    out = np.zeros(4 * k + 2, dtype=np.float64)
    for i in range(k):
        if i < poles.size:
            out[i] = _logit(float(np.abs(poles[i])), RHO_MAX)
            out[k + i] = float(np.angle(poles[i]))
        else:
            out[i] = -12.0  # radius ~ 0 (inert pole)
        if i < zeros.size:
            out[2 * k + i] = _logit(float(np.abs(zeros[i])), _ZERO_SCALE)
            out[3 * k + i] = float(np.angle(zeros[i]))
        else:
            out[2 * k + i] = -12.0
    out[4 * k] = float(kernel.gain.real)
    out[4 * k + 1] = float(kernel.gain.imag)
    return out


def _unconstrained_to_threshold(u: float) -> float:
    return -_softplus(float(u))


def _threshold_to_unconstrained(b: float) -> float:
    # inverse of -softplus: b <= 0 -> u with softplus(u) = -b
    t = -float(b)
    if t < 1e-9:
        return -12.0
    return float(np.log(np.expm1(t)))
