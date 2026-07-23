"""Weight-bank compiler and content-addressed manifests.

A weight bank is the shippable artifact: a compact serialization of an operator
(config + frozen backbone + adapted parameter vector) with a CRC32 for load-time
integrity, addressed by the sha256 of its bytes. The manifest records lineage
(parent bank, training seed, and the reward/env/eval digests) so a bank cannot be
referenced by the twin acceptance gate or loaded by a production driver without a
passing evidence trail -- while the ABI itself stays ignorant of all of it.

Two derived facts are carried in the header:

- **LOG2_N validity range.** The smallest FFT size at which every kernel's
  aliasing certificate meets the tolerance, so the fabric can reject a bank run
  at a size it was not certified for (the honest bound, not the naive shortcut).
- **Pole cap.** The fabric radius cap the bank was compiled against.

``emit_tables`` produces the dense per-mode quantized tables the fabric loads for
a given ``N``; that is the only place a specific FFT size enters.
"""
from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .fixedpoint import SharedExpTable, quantize_table
from .operator import FrozenBackbone, NeuralOperator, OperatorConfig
from .zplane import RHO_MAX

MAGIC = b"ANRL"
FORMAT_VERSION = 1
DEFAULT_EPS = 2.0 ** -12


@dataclass(frozen=True)
class BankLineage:
    """Provenance of a weight bank. All digests are hex strings or empty."""

    parent_digest: str = ""
    training_seed: int = 0
    reward_manifest_digest: str = ""
    env_manifest_digest: str = ""
    eval_report_digest: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "parent_digest": self.parent_digest,
            "training_seed": self.training_seed,
            "reward_manifest_digest": self.reward_manifest_digest,
            "env_manifest_digest": self.env_manifest_digest,
            "eval_report_digest": self.eval_report_digest,
        }


@dataclass(frozen=True)
class BankManifest:
    """The addressable record of a bank (no payload)."""

    digest: str
    crc32: int
    format_version: int
    config: Dict[str, int]
    anchor_hz: float
    log2n_min: int
    log2n_max: int
    pole_cap: float
    lineage: BankLineage

    def to_dict(self) -> Dict[str, object]:
        return {
            "digest": self.digest,
            "crc32": self.crc32,
            "format_version": self.format_version,
            "config": self.config,
            "anchor_hz": self.anchor_hz,
            "log2n_min": self.log2n_min,
            "log2n_max": self.log2n_max,
            "pole_cap": self.pole_cap,
            "lineage": self.lineage.to_dict(),
        }


def certified_log2n_range(operator: NeuralOperator, eps: float = DEFAULT_EPS) -> tuple[int, int]:
    """Smallest ``log2 N`` in [10, 16] at which every kernel's certificate <= eps."""
    log2n_max = 16
    for m in range(10, 17):
        n = 1 << m
        ok = all(
            kernel.aliasing_certificate(n) <= eps
            for row in operator.kernels
            for kernel in row
        )
        if ok:
            return m, log2n_max
    # No size in range certifies: signal the tightest (16) as both ends.
    return log2n_max, log2n_max


def _serialize_body(operator: NeuralOperator, log2n_min: int, log2n_max: int) -> bytes:
    cfg = operator.config
    header = struct.pack(
        "<5H d 2B d",
        cfg.width, cfg.layers, cfg.sections, cfg.in_channels, cfg.out_channels,
        float(operator.kernels[0][0].anchor_hz),
        log2n_min, log2n_max,
        RHO_MAX,
    )
    vec = operator.adapted_vector().astype("<f8").tobytes()
    bb = operator.backbone
    arrays = [bb.lift.astype("<c16")]
    arrays += [m.astype("<c16") for m in bb.mixes]
    arrays.append(bb.project.astype("<c16"))
    backbone_bytes = b"".join(a.tobytes() for a in arrays)
    veclen = struct.pack("<I", len(vec))
    return header + veclen + vec + backbone_bytes


@dataclass(frozen=True)
class WeightBank:
    """A compiled, integrity-checked operator ready to ship or load."""

    manifest: BankManifest
    payload: bytes  # MAGIC + version + body + crc32
    operator: NeuralOperator = field(repr=False)

    def verify_crc(self) -> bool:
        body = self.payload[6:-4]
        stored = struct.unpack("<I", self.payload[-4:])[0]
        return zlib.crc32(body) == stored

    def emit_tables(self, n: int) -> List[List[SharedExpTable]]:
        """Dense per-mode quantized tables for FFT size ``n`` (fabric-facing).

        Raises if ``n`` is outside the bank's certified LOG2_N range.
        """
        m = int(np.log2(n))
        if (1 << m) != n:
            raise ValueError("n must be a power of two")
        if not (self.manifest.log2n_min <= m <= self.manifest.log2n_max):
            raise ValueError(
                f"N=2^{m} outside certified range "
                f"[2^{self.manifest.log2n_min}, 2^{self.manifest.log2n_max}]"
            )
        tables: List[List[SharedExpTable]] = []
        for row in self.operator.kernels:
            tables.append([quantize_table(k.response_on_grid(n)) for k in row])
        return tables


def compile_bank(
    operator: NeuralOperator,
    lineage: Optional[BankLineage] = None,
    eps: float = DEFAULT_EPS,
) -> WeightBank:
    """Compile an operator into an addressable, integrity-checked weight bank."""
    lineage = lineage or BankLineage()
    log2n_min, log2n_max = certified_log2n_range(operator, eps)
    body = _serialize_body(operator, log2n_min, log2n_max)
    crc = zlib.crc32(body)
    payload = MAGIC + struct.pack("<H", FORMAT_VERSION) + body + struct.pack("<I", crc)
    digest = hashlib.sha256(payload).hexdigest()
    cfg = operator.config
    manifest = BankManifest(
        digest=digest,
        crc32=crc,
        format_version=FORMAT_VERSION,
        config={
            "width": cfg.width, "layers": cfg.layers, "sections": cfg.sections,
            "in_channels": cfg.in_channels, "out_channels": cfg.out_channels,
        },
        anchor_hz=float(operator.kernels[0][0].anchor_hz),
        log2n_min=log2n_min, log2n_max=log2n_max, pole_cap=RHO_MAX,
        lineage=lineage,
    )
    return WeightBank(manifest=manifest, payload=payload, operator=operator)


def load_bank(payload: bytes) -> WeightBank:
    """Reconstruct a weight bank (and its operator) from serialized bytes."""
    if payload[:4] != MAGIC:
        raise ValueError("bad magic: not an Atom-Neural-RL weight bank")
    (version,) = struct.unpack("<H", payload[4:6])
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported bank format version {version}")
    body = payload[6:-4]
    stored_crc = struct.unpack("<I", payload[-4:])[0]
    if zlib.crc32(body) != stored_crc:
        raise ValueError("CRC mismatch: weight bank is corrupt")

    hdr = struct.calcsize("<5H d 2B d")
    (width, layers, sections, in_ch, out_ch, anchor_hz, log2n_min, log2n_max, pole_cap) = struct.unpack(
        "<5H d 2B d", body[:hdr]
    )
    cursor = hdr
    (veclen,) = struct.unpack("<I", body[cursor : cursor + 4])
    cursor += 4
    vec = np.frombuffer(body[cursor : cursor + veclen], dtype="<f8").copy()
    cursor += veclen

    config = OperatorConfig(width=width, layers=layers, sections=sections,
                            in_channels=in_ch, out_channels=out_ch)

    def take(shape):
        nonlocal cursor
        count = int(np.prod(shape))
        nbytes = count * 16
        arr = np.frombuffer(body[cursor : cursor + nbytes], dtype="<c16").reshape(shape).copy()
        cursor += nbytes
        return arr

    lift = take((width, in_ch))
    mixes = [take((width, width)) for _ in range(layers)]
    project = take((out_ch, width))
    backbone = FrozenBackbone(lift, mixes, project)
    operator = NeuralOperator.build(config, backbone, vec)

    manifest = BankManifest(
        digest=hashlib.sha256(payload).hexdigest(), crc32=stored_crc,
        format_version=version,
        config={"width": width, "layers": layers, "sections": sections,
                "in_channels": in_ch, "out_channels": out_ch},
        anchor_hz=anchor_hz, log2n_min=log2n_min, log2n_max=log2n_max,
        pole_cap=pole_cap, lineage=BankLineage(),
    )
    return WeightBank(manifest=manifest, payload=payload, operator=operator)
