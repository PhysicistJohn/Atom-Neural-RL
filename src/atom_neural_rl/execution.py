"""ABI-faithful fixed-point execution -- the loop closed through the contract.

``AbiExecutor`` runs an operator the way the fabric (and the twin device model)
will: it loads a weight bank (CRC-checked), evaluates each kernel per current N
and sample rate, quantizes the response to a Q1.15 shared-exponent table, and
applies it with the fixed-point complex multiply and exponent-compensated
modReLU. The shared FFT core is modelled in float, because its own fixed-point
behaviour is the twin's separate contract (int32/Q2.30); what this models is the
operator-*added* datapath, which is where sim-to-hardware transfer is at risk.

It implements the operator interface (``config``, ``forward``, ``adapted_vector``,
``with_adapted_vector``), so the *same* CMA-ES training loop that runs on the
float operator runs unchanged through the fixed-point ABI. If the coherence
reward still improves when computed on the quantized output, the loop closes
through the contract -- which is the meaningful "runs on the twin" claim, since
the twin implements this same ABI.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .fixedpoint import fixed_point_spectral_multiply, modrelu_fixed, quantize_table
from .operator import NeuralOperator, OperatorConfig


class AbiExecutor:
    """Fixed-point, ABI-faithful execution wrapper around an operator."""

    def __init__(self, operator: NeuralOperator, bank=None) -> None:
        self.operator = operator
        self.config: OperatorConfig = operator.config
        self.bank = bank
        # Latched per result, as the ABI's RESULT_WEIGHT_CRC would be.
        self.result_weight_crc: Optional[int] = bank.manifest.crc32 if bank is not None else None

    # -- construction / the operator interface ---------------------------
    @classmethod
    def from_operator(cls, operator: NeuralOperator) -> "AbiExecutor":
        return cls(operator)

    @classmethod
    def from_bank(cls, bank) -> "AbiExecutor":
        """Load a weight bank through the ABI: CRC gate first (BANK_INTEGRITY)."""
        if not bank.verify_crc():
            raise ValueError("BANK_INTEGRITY: weight-bank CRC check failed")
        return cls(bank.operator, bank)

    def adapted_vector(self) -> np.ndarray:
        return self.operator.adapted_vector()

    def with_adapted_vector(self, vector: np.ndarray) -> "AbiExecutor":
        return AbiExecutor(self.operator.with_adapted_vector(vector))

    # -- fixed-point block execution -------------------------------------
    def forward(self, iq: np.ndarray, fs_hz: float) -> np.ndarray:
        """Execute one capture block in the fixed-point ABI datapath.

        Mirrors :meth:`NeuralOperator.forward` but with quantized weight tables,
        the fixed-point complex multiply, and exponent-compensated modReLU.
        """
        op = self.operator
        cfg = op.config
        iq = np.asarray(iq, dtype=np.complex128)
        if iq.ndim != 2 or iq.shape[0] != cfg.in_channels:
            raise ValueError(f"iq must be ({cfg.in_channels}, N)")
        n = iq.shape[1]
        # LOG2_N validity, as the engine would reject at START (BAD_OP_CONFIG).
        if (n & (n - 1)) != 0:
            raise ValueError("N must be a power of two")
        if self.bank is not None:
            m = int(np.log2(n))
            if not (self.bank.manifest.log2n_min <= m <= self.bank.manifest.log2n_max):
                raise ValueError(f"BAD_OP_CONFIG: N=2^{m} outside bank's certified range")

        v = np.einsum("wc,cn->wn", op.backbone.lift, iq)
        for layer in range(cfg.layers):
            spectral = np.empty_like(v)
            for w in range(cfg.width):
                kernel = op.kernels[layer][w]
                if fs_hz != kernel.anchor_hz:
                    kernel = kernel.transport(fs_hz)
                table = quantize_table(kernel.response_on_grid(n))  # the loaded weights
                vf = np.fft.fft(v[w])
                yf = fixed_point_spectral_multiply(table, vf)
                spectral[w] = np.fft.ifft(yf)
            mixed = np.einsum("ij,jn->in", op.backbone.mixes[layer], v)
            pre = mixed + spectral
            for w in range(cfg.width):
                v[w] = modrelu_fixed(pre[w], float(op.thresholds[layer, w]))
        return np.einsum("cw,wn->cn", op.backbone.project, v)
