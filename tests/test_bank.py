"""Weight-bank compiler: round-trip, integrity, N-range, digest stability."""
from __future__ import annotations

import unittest

import numpy as np

from atom_neural_rl.bank import (
    BankLineage,
    certified_log2n_range,
    compile_bank,
    load_bank,
)
from atom_neural_rl.operator import NeuralOperator, OperatorConfig
from atom_neural_rl.zplane import F0_HZ


def _trained_operator(seed: int = 0) -> NeuralOperator:
    cfg = OperatorConfig.diagonal_for_channels(2, sections=8)
    op = NeuralOperator.warm_start(cfg)
    rng = np.random.default_rng(seed)
    return op.with_adapted_vector(rng.standard_normal(cfg.adapted_dim) * 0.3)


class RoundTrip(unittest.TestCase):
    def test_bank_reconstructs_identical_operator(self) -> None:
        op = _trained_operator(1)
        bank = compile_bank(op)
        restored = load_bank(bank.payload).operator
        rng = np.random.default_rng(9)
        iq = (rng.standard_normal((2, 512)) + 1j * rng.standard_normal((2, 512))) / np.sqrt(2)
        np.testing.assert_allclose(op.forward(iq, F0_HZ), restored.forward(iq, F0_HZ), atol=1e-9)

    def test_digest_is_deterministic(self) -> None:
        op = _trained_operator(2)
        a = compile_bank(op).manifest.digest
        b = compile_bank(op).manifest.digest
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)

    def test_different_operators_have_different_digests(self) -> None:
        self.assertNotEqual(
            compile_bank(_trained_operator(3)).manifest.digest,
            compile_bank(_trained_operator(4)).manifest.digest,
        )


class Integrity(unittest.TestCase):
    def test_crc_detects_corruption(self) -> None:
        bank = compile_bank(_trained_operator(5))
        self.assertTrue(bank.verify_crc())
        corrupt = bytearray(bank.payload)
        corrupt[20] ^= 0xFF  # flip a payload byte
        with self.assertRaises(ValueError):
            load_bank(bytes(corrupt))

    def test_bad_magic_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_bank(b"XXXX" + b"\x00" * 40)


class Certification(unittest.TestCase):
    def test_identity_certifies_at_smallest_size(self) -> None:
        op = NeuralOperator.warm_start(OperatorConfig.diagonal_for_channels(1, sections=8))
        lo, hi = certified_log2n_range(op)
        self.assertEqual(lo, 10)
        self.assertEqual(hi, 16)

    def test_emit_tables_respects_certified_range(self) -> None:
        op = _trained_operator(6)
        bank = compile_bank(op)
        n = 1 << bank.manifest.log2n_max
        tables = bank.emit_tables(n)
        self.assertEqual(len(tables), op.config.layers)
        self.assertEqual(len(tables[0]), op.config.width)
        self.assertEqual(tables[0][0].size, n)

    def test_emit_tables_rejects_uncertified_size(self) -> None:
        op = _trained_operator(7)
        bank = compile_bank(op)
        if bank.manifest.log2n_min > 10:
            with self.assertRaises(ValueError):
                bank.emit_tables(1 << (bank.manifest.log2n_min - 1))
        else:
            self.skipTest("bank certified down to 2^10; no smaller size to reject")


class Lineage(unittest.TestCase):
    def test_lineage_is_carried_in_manifest(self) -> None:
        lineage = BankLineage(parent_digest="abc123", training_seed=42,
                              eval_report_digest="deadbeef")
        bank = compile_bank(_trained_operator(8), lineage=lineage)
        d = bank.manifest.to_dict()
        self.assertEqual(d["lineage"]["parent_digest"], "abc123")
        self.assertEqual(d["lineage"]["training_seed"], 42)


if __name__ == "__main__":
    unittest.main()
