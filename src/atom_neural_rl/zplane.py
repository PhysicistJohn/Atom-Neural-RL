"""Rational z-plane kernels, master-rate transport, and the invariance certificate.

This module is the mathematical core and the place where "true z-plane
invariance" is made precise and checkable.

Representation
--------------
A kernel is a rational transfer function in product (gain / pole / zero) form:

    H(z) = g * prod_i (1 - q_i z^-1) / prod_j (1 - p_j z^-1)

with free *complex* poles ``p_j`` and zeros ``q_i`` (no conjugate-pair
constraint, so single-sided baseband behaviour is expressible) and a complex
gain ``g``. Poles are constrained to ``|p_j| <= RHO_MAX`` for stability and for
a bounded discretization error.

Product form is the normative representation here because the optimizer is
CMA-ES (gradient-free): the identity kernel is the *exact* ``g = 1, zeros ==
poles`` cancellation, with no dead-gradient pathology. The residue (partial
fraction) form is provided by :func:`residues` for the certificate and for a
future JAX gradient-pretraining backend.

The invariance statement, precisely
-----------------------------------
The learned object is a single rational function on the z-plane, anchored at the
master rate ``F0_HZ``. Two facts, both tested:

1. **Discretization (N) invariance is exact for the response.** The frequency
   response ``H(e^{jw})`` is a closed-form function of normalized frequency
   ``w``; sampling it at ``N`` grid points and reading it at a fixed physical
   frequency gives the identical value for every ``N`` (to floating point). See
   :func:`evaluate` and ``test_invariance``.

2. **Applying the kernel by N-point circular convolution has a certified
   error.** Frequency sampling realizes circular convolution with the
   time-aliased impulse response; the deviation from the true (aperiodic)
   filtering is bounded by

       eps(N) <= 2 * R * rho^N / (1 - rho),   R = sum_j |r_j|, rho = max_j |p_j|

   where ``r_j`` are the residues. See :func:`aliasing_certificate`. This is the
   honest bound; the naive ``rho^N <= eps`` rule drops the ``1/(1-rho)``
   prefactor (~124x at ``N = 2^10``) and leaves ``R`` unconstrained, so both the
   prefactor and a certified ``R`` are carried here.

3. **Rate handling is master-rate anchoring with no fallback.** Every kernel is
   anchored at ``F0_HZ`` (the AD9361 maximum). Deploying at ``fs <= F0_HZ`` maps
   each pole ``p -> p^(F0_HZ/fs)``; since the exponent is ``>= 1`` and
   ``|p| < 1``, poles only ever move *inward*, so stability and the certificate
   improve automatically and the radial projection can never fire in
   deployment. Invariance on the z-plane is, necessarily, covariance in Hz: the
   response at a fixed *normalized* frequency is preserved by transport. See
   :func:`transport` and ``test_invariance``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# The AD9361 master sample rate. Every kernel is anchored here; no legal capture
# rate exceeds it, which is what makes transport strictly stability-improving.
F0_HZ: float = 61_440_000.0

# Fabric pole-radius cap. Coupled to the interpolation grid M = 4096 and the
# 2 MB weight-bank size (see the design spec); 1 - 2^-7 exactly.
RHO_MAX: float = 1.0 - 2.0 ** -7  # 0.9921875


@dataclass(frozen=True)
class RationalKernel:
    """A scalar rational kernel in product form, anchored at :data:`F0_HZ`.

    ``gain`` is complex; ``poles`` and ``zeros`` are complex arrays of equal or
    differing length. The kernel is immutable; transforms return new kernels.
    """

    gain: complex
    poles: np.ndarray
    zeros: np.ndarray
    anchor_hz: float = F0_HZ

    def __post_init__(self) -> None:
        object.__setattr__(self, "poles", np.asarray(self.poles, dtype=np.complex128))
        object.__setattr__(self, "zeros", np.asarray(self.zeros, dtype=np.complex128))
        if self.poles.ndim != 1 or self.zeros.ndim != 1:
            raise ValueError("poles and zeros must be 1-D")
        if np.any(np.abs(self.poles) >= 1.0):
            raise ValueError("all poles must lie strictly inside the unit disk")

    # -- construction -----------------------------------------------------
    @staticmethod
    def identity() -> "RationalKernel":
        """The exact identity ``H(z) == 1`` (empty pole/zero sets, unit gain)."""
        empty = np.zeros(0, dtype=np.complex128)
        return RationalKernel(1.0 + 0j, empty, empty)

    @property
    def order(self) -> int:
        return int(max(self.poles.size, self.zeros.size))

    @property
    def pole_radius(self) -> float:
        """``max |p_j|``; 0.0 for a kernel with no poles."""
        return float(np.max(np.abs(self.poles))) if self.poles.size else 0.0

    # -- evaluation -------------------------------------------------------
    def evaluate(self, omega: np.ndarray) -> np.ndarray:
        """Frequency response ``H(e^{j omega})`` at normalized angular frequencies.

        ``omega`` is in radians/sample. The result is a closed-form function of
        ``omega`` alone, which is exactly why sampling at any ``N`` is
        self-consistent.
        """
        z_inv = np.exp(-1j * np.asarray(omega, dtype=np.float64))
        num = np.ones_like(z_inv, dtype=np.complex128)
        for q in self.zeros:
            num = num * (1.0 - q * z_inv)
        den = np.ones_like(z_inv, dtype=np.complex128)
        for p in self.poles:
            den = den * (1.0 - p * z_inv)
        return self.gain * num / den

    def response_on_grid(self, n: int) -> np.ndarray:
        """``H`` sampled on the ``n``-point DFT grid, natural bin order."""
        k = np.arange(n)
        return self.evaluate(2.0 * np.pi * k / n)

    def impulse_response(self, length: int) -> np.ndarray:
        """The causal impulse response ``h[0..length-1]`` (for the certificate)."""
        # Long-division of the product-form rational function.
        num = _poly_from_roots_zinv(self.zeros) * self.gain
        den = _poly_from_roots_zinv(self.poles)
        h = np.zeros(length, dtype=np.complex128)
        # den[0] == 1 by construction of _poly_from_roots_zinv.
        for n in range(length):
            acc = num[n] if n < num.size else 0.0 + 0j
            for k in range(1, min(n, den.size - 1) + 1):
                acc -= den[k] * h[n - k]
            h[n] = acc
        return h

    # -- stability + transport -------------------------------------------
    def project_stable(self, rho_max: float = RHO_MAX) -> "RationalKernel":
        """Radially project poles onto ``|p| <= rho_max`` (no-op if already inside)."""
        poles = self.poles.copy()
        radii = np.abs(poles)
        over = radii > rho_max
        if np.any(over):
            poles[over] = poles[over] / radii[over] * rho_max
        return RationalKernel(self.gain, poles, self.zeros, self.anchor_hz)

    def transport(self, fs_hz: float) -> "RationalKernel":
        """Matched-z re-anchor from the master rate to deployment rate ``fs_hz``.

        Each singularity maps ``s -> s^(F0/fs)`` with ``F0/fs >= 1`` (no capture
        exceeds the master rate), so poles move strictly inward. The returned
        kernel carries ``anchor_hz = fs_hz``. Gain is preserved; downstream
        table compilation applies the N-independent normalization.
        """
        if fs_hz <= 0.0 or fs_hz > self.anchor_hz + 1e-6:
            raise ValueError(
                f"deployment rate {fs_hz} must be in (0, anchor={self.anchor_hz}]"
            )
        exponent = self.anchor_hz / fs_hz
        poles = _principal_power(self.poles, exponent)
        zeros = _principal_power(self.zeros, exponent)
        return RationalKernel(self.gain, poles, zeros, fs_hz)

    # -- residues + certificate ------------------------------------------
    def residues(self) -> tuple[np.ndarray, np.ndarray, complex]:
        """Partial-fraction residues: ``(r_j, p_j, c0)`` with ``H = c0 + sum r_j/(1-p_j z^-1)``.

        Assumes distinct poles and ``deg(num) <= deg(den)`` in ``z^-1`` (true for
        the kernels this package produces). Repeated poles are perturbed by a
        tiny amount so the Heaviside cover-up rule applies; this only affects the
        certificate, never the response.
        """
        poles = _split_repeated(self.poles)
        num = _poly_from_roots_zinv(self.zeros) * self.gain
        den = _poly_from_roots_zinv(poles)
        # Direct term c0: value of H as z^-1 -> inf is num_top/den_top ratio only
        # when degrees are equal; for strictly proper H, c0 = 0. We compute c0 as
        # the quotient of leading coefficients when degrees match.
        deg_num = num.size - 1
        deg_den = den.size - 1
        if deg_num < deg_den:
            c0 = 0.0 + 0j
        else:
            c0 = num[-1] / den[-1] if den[-1] != 0 else 0.0 + 0j
        r = np.zeros(poles.size, dtype=np.complex128)
        for j, p in enumerate(poles):
            # Residue of H(z) at pole p in the z^-1 partial fraction:
            #   r_j = H_proper(z) * (1 - p z^-1) evaluated at z^-1 = 1/p.
            zinv = 1.0 / p
            num_val = np.polyval(num[::-1], zinv)
            den_others = 1.0 + 0j
            for k, pk in enumerate(poles):
                if k == j:
                    continue
                den_others *= (1.0 - pk * zinv)
            r[j] = num_val / den_others if den_others != 0 else 0.0 + 0j
        return r, poles, c0

    def residue_norm(self) -> float:
        """``R = sum_j |r_j|`` -- the certified quantity in the aliasing bound."""
        r, _poles, _c0 = self.residues()
        return float(np.sum(np.abs(r)))

    def aliasing_certificate(self, n: int) -> float:
        """Certified bound on N-point circular vs true aperiodic filtering.

            eps(N) <= 2 R rho^N / (1 - rho)

        Returns ``0.0`` for a kernel with no poles (an FIR/identity kernel is
        realized exactly by circular convolution once ``N >= len(h)``).
        """
        rho = self.pole_radius
        if rho == 0.0:
            return 0.0
        R = self.residue_norm()
        return float(2.0 * R * rho ** n / (1.0 - rho))

    def max_stable_radius_for(self, n: int, eps: float) -> float:
        """Largest ``rho`` whose certificate meets ``eps`` at size ``n``.

        Solves ``2 R rho^N / (1 - rho) <= eps`` for ``rho`` by bisection, using
        this kernel's residue norm ``R``. Used by bank validation to pick the
        declared ``LOG2_N`` validity range honestly rather than via the naive
        ``eps^(1/N)`` shortcut.
        """
        R = max(self.residue_norm(), 1e-12)
        lo, hi = 0.0, RHO_MAX
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            bound = 2.0 * R * mid ** n / (1.0 - mid)
            if bound <= eps:
                lo = mid
            else:
                hi = mid
        return lo


# ---------------------------------------------------------------------------
# polynomial helpers (in z^-1)
# ---------------------------------------------------------------------------
def _poly_from_roots_zinv(roots: np.ndarray) -> np.ndarray:
    """Coefficients (ascending powers of z^-1) of ``prod_i (1 - root_i z^-1)``.

    Returns ``[1]`` for an empty root set. ``coeff[0] == 1`` always.
    """
    coeff = np.array([1.0 + 0j], dtype=np.complex128)
    for r in roots:
        # multiply by (1 - r z^-1): convolution with [1, -r]
        coeff = np.convolve(coeff, np.array([1.0, -r], dtype=np.complex128))
    return coeff


def _principal_power(values: np.ndarray, exponent: float) -> np.ndarray:
    """``v ** exponent`` via the principal branch, radius/angle separated.

    Keeps the angle continuous (``r^a e^{i a theta}``) so a resonance at angle
    ``theta`` maps to angle ``a*theta`` -- the matched-z frequency dilation.
    """
    values = np.asarray(values, dtype=np.complex128)
    if values.size == 0:
        return values
    radius = np.abs(values)
    angle = np.angle(values)
    new_radius = np.where(radius > 0, radius ** exponent, 0.0)
    return new_radius * np.exp(1j * exponent * angle)


def _split_repeated(poles: np.ndarray, jitter: float = 1e-7) -> np.ndarray:
    """Perturb coincident poles apart so the cover-up residue rule applies.

    Deterministic (no RNG): equal poles are fanned out along a fixed small
    angular offset. Affects only the certificate estimate, never the response.
    """
    poles = poles.astype(np.complex128).copy()
    seen: dict[complex, int] = {}
    for i, p in enumerate(poles):
        key = complex(round(p.real, 12), round(p.imag, 12))
        count = seen.get(key, 0)
        if count:
            poles[i] = p * (1.0 + jitter * count) * np.exp(1j * jitter * count)
        seen[key] = count + 1
    return poles


def make_resonator(radius: float, angle: float, gain: complex = 1.0 + 0j) -> RationalKernel:
    """A single complex-conjugate-free resonator: one pole, no zeros.

    Convenience constructor for tests and simple excision/enhancement kernels.
    """
    if not 0.0 <= radius < 1.0:
        raise ValueError("radius must be in [0, 1)")
    pole = np.array([radius * np.exp(1j * angle)], dtype=np.complex128)
    return RationalKernel(gain, pole, np.zeros(0, dtype=np.complex128))


def make_notch(radius: float, angle: float, depth: float = 0.98) -> RationalKernel:
    """A single-sided notch: a zero near the unit circle with a stabilizing pole.

    ``depth`` in [0,1) sets how deep the null is (zero radius = ``depth``); the
    pole sits at ``depth * 0.9`` on the same angle to keep the notch narrow and
    the kernel stable. Single-sided by construction (no conjugate zero), which is
    the behaviour conjugate-pairing would forbid.
    """
    z = np.array([depth * np.exp(1j * angle)], dtype=np.complex128)
    p = np.array([radius * np.exp(1j * angle)], dtype=np.complex128)
    return RationalKernel(1.0 + 0j, p, z)
