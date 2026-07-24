# Atom-Neural-RL

A z-plane invariant complex Fourier neural operator for the NeptuneSDR/P210,
learned under reinforcement learning. This repository owns the operator
mathematics (JAX reference and bit-exact fixed-point model), the rational
z-plane kernel tooling, the weight-bank compiler and content-addressed bank
manifests, the CMA-ES training harness, the host-side reward stack, the
SignalLab gym bindings, and the evaluation gates.

The exact repository identity is `Atom-Neural-RL`, published at
<https://github.com/PhysicistJohn/Atom-Neural-RL>.

## The invariance contract

The learned object is a single rational function on the z-plane, anchored at
the master rate 61.44 MSPS. Deployment at any FFT size (2^10..2^16) and any
sample rate at or below the master rate evaluates that same object through one
deterministic map: matched-z transport (poles only ever move inward), unit
circle evaluation, quantization. The transport is trained through, so
training, simulation, twin, and hardware execute the identical pipeline.
Discretization error is certified per bank; there are no fallbacks.

## Ownership boundary

The PL register contract lives in Atom-NeptuneSDR-Firmware (interface JSON v2,
a superset of v1). The QEMU device model and acceptance gates live in
Atom-NeptuneSDR-Twin. SignalLab is the training environment; Atom-Classifier
and Atom-DSP supply reward judges. Repositories couple only through versioned
JSON contracts and content-addressed weight-bank manifests, never through
imported code. The fabric never sees poles, tasks, or rewards: it multiplies
tables.

## Status

Phase P1: design spec signed off; reference implementation in progress. The
design document (three lanes, independently adversarially verified) governs
this repository.

## Reproduce

Requirements: Python >= 3.9. numpy is the only runtime dependency, and it is
installed for you by the editable install below.

```
python3 -m pip install -e .                             # install (numpy only)
PYTHONPATH=src python3 -m unittest discover -s tests    # full suite (114 tests, ~1.5 min)
PYTHONPATH=src python3 -m atom_neural_rl.cli verify-invariance
scripts/check.sh                                        # the CI source gate
```

`golden.py` is the normative fixed-point reference; its pinned vector digests
are reproduced bit-for-bit by the C twin core, the RTL, and the QEMU operator
device (see the operator chain in
[`Atom-NeptuneSDR-Twin/cosim/REPRODUCE.md`](https://github.com/PhysicistJohn/Atom-NeptuneSDR-Twin/blob/main/cosim/REPRODUCE.md)).
Design: [`docs/DESIGN.md`](docs/DESIGN.md); day-one hardware runbook:
[`docs/DAY_ONE.md`](docs/DAY_ONE.md).
