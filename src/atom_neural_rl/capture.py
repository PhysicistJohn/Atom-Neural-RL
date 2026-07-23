"""Board-faithful IQ capture ingestion -- the seam to real hardware.

Encodes and decodes the P210 wideband capture format exactly as the spec defines
it (``specs/p210-firmware-interface-v2.json`` ``wideband_capture``): signed IQ16
little-endian, time-major channel-interleaved, 12 significant bits with
full-scale code 2048. The same decode path serves a real ``iiod`` byte stream, a
capture file, or the sim gym as a stand-in, so the downstream host-side operator
runs identically on board, twin, and sim data -- pointing at the real board is a
capture-source swap, not new code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# adc_full_scale_code from the spec: 12-bit signed, so codes span [-2048, 2047].
IQ16_FULL_SCALE = 2048
IIOD_PORT = 30431
NSFT_PORT = 30432


def encode_iq16(iq: np.ndarray, full_scale: int = IQ16_FULL_SCALE) -> bytes:
    """Encode ``(channels, N)`` complex baseband to the board's IQ16 byte layout.

    Time-major channel-interleaved: for each sample n, ``I0 Q0 I1 Q1 ...`` with
    each component a signed 16-bit little-endian code. Values are scaled by
    ``full_scale`` and clipped to the int16 range.
    """
    iq = np.asarray(iq, dtype=np.complex128)
    if iq.ndim != 2:
        raise ValueError("iq must be (channels, N)")
    channels, n = iq.shape
    inter = np.empty((n, channels, 2), dtype="<i2")
    for c in range(channels):
        inter[:, c, 0] = np.clip(np.round(iq[c].real * full_scale), -32768, 32767)
        inter[:, c, 1] = np.clip(np.round(iq[c].imag * full_scale), -32768, 32767)
    return inter.tobytes()


def decode_iq16(data: bytes, channels: int, full_scale: int = IQ16_FULL_SCALE) -> np.ndarray:
    """Decode the board's IQ16 byte layout to ``(channels, N)`` complex baseband."""
    arr = np.frombuffer(data, dtype="<i2")
    per_sample = 2 * channels
    n = arr.size // per_sample
    inter = arr[: n * per_sample].reshape(n, channels, 2).astype(np.float64)
    iq = (inter[:, :, 0] + 1j * inter[:, :, 1]) / full_scale
    return iq.T.copy()  # (channels, N)


@dataclass(frozen=True)
class Capture:
    """One ingested capture: IQ plus the metadata the operator needs."""

    iq: np.ndarray          # (channels, N) complex
    fs_hz: float
    sps: Optional[int] = None       # known emitter oversampling, if any
    clean: Optional[np.ndarray] = None  # ground truth, only in sim/twin rehearsal

    @property
    def channels(self) -> int:
        return int(self.iq.shape[0])


class CaptureSource:
    """Interface for a source of captures. Real and sim sources are interchangeable."""

    def capture(self, n_samples: int) -> Capture:  # pragma: no cover - interface
        raise NotImplementedError


class GymCaptureSource(CaptureSource):
    """Sim/twin stand-in: draws captures from the gym, carrying ground truth so
    the rehearsal can measure the *true* quality the board cannot report."""

    def __init__(self, gym, seed: int = 0) -> None:
        self.gym = gym
        self._rng = np.random.default_rng(seed)

    def capture(self, n_samples: int) -> Capture:
        spec = self.gym.sample_spec(self._rng, n_samples=n_samples)
        ep = self.gym.realize(spec)
        return Capture(iq=ep.observed, fs_hz=spec.fs_hz, sps=spec.profile.sps, clean=ep.clean)


class BytesCaptureSource(CaptureSource):
    """Ingests raw IQ16 bytes (a file, or a buffered iiod read) -- the real path.

    The board provides no ground truth, so ``clean`` is ``None``; quality is
    reported blind. This is the exact code that runs against hardware.
    """

    def __init__(self, data: bytes, channels: int, fs_hz: float, sps: Optional[int] = None) -> None:
        self._iq = decode_iq16(data, channels)
        self.fs_hz = fs_hz
        self.sps = sps

    def capture(self, n_samples: int) -> Capture:
        return Capture(iq=self._iq[:, :n_samples], fs_hz=self.fs_hz, sps=self.sps, clean=None)

# A live libiio/iiod TCP source (host=board, port=IIOD_PORT) is a drop-in
# CaptureSource: connect, read the interleaved IQ16 buffer, hand the bytes to
# decode_iq16. It is intentionally not implemented here because it needs libiio
# and a reachable board; the interface above is what it plugs into.
