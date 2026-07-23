#!/usr/bin/env bash
# Standalone source gate: the full unittest suite with warnings promoted to
# errors, so the spurious-warning-clean invariant is enforced in CI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

python3 -W error::RuntimeWarning -m unittest discover -s tests -v

echo "ATOM_NEURAL_RL_SOURCE_GATE PASS"
