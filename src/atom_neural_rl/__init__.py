"""Atom-Neural-RL: a z-plane invariant complex Fourier neural operator, learned under RL.

The public surface is deliberately small and layered so each concern is testable
in isolation and the invariance claims are demonstrable, not asserted:

- ``zplane``      the rational z-plane kernel, master-rate transport, and the
                  discretization-invariance certificate.
- ``operator``    the complex FNO (lift, diagonal spectral, pointwise mixing,
                  modReLU) evaluated block-wise over IQ captures.
- ``fixedpoint``  the bit-exact fixed-point datapath model the fabric realizes.
- ``bank``        the weight-bank compiler and content-addressed manifests.
- ``waveforms`` / ``channel`` / ``gym``  the deterministic training environment.
- ``recovery``    the blind-recovery reward judge (Python port of the classifier).
- ``reward``      the host-side paired-differential reward stack (v1 signal quality).
- ``cma``         the CMA-ES optimizer over the small kernel parameter vector.
- ``gates``       the G1-G4 promotion gates.

Nothing in this package touches hardware. The fabric contract lives in
Atom-NeptuneSDR-Firmware; the executable twin lives in Atom-NeptuneSDR-Twin.
"""
from __future__ import annotations

from .version import __version__

__all__ = ["__version__"]
