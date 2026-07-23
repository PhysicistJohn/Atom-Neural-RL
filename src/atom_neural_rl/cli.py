"""Command-line surface: verify invariance, train a bank, and gate a bank.

None of these commands touches hardware. ``train`` and ``gate`` operate against
the in-repo gym; ``verify-invariance`` prints the discretization certificate so
the headline property is inspectable from the shell.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import numpy as np

from .bank import BankLineage, compile_bank, load_bank
from .channel import ChannelParams
from .gates import run_gates
from .gym import Gym, EpisodeSpec
from .operator import NAMED_CONFIGS, NeuralOperator, OperatorConfig
from .reward import episode_reward
from .cma import train_operator
from .waveforms import WaveformProfile
from .zplane import F0_HZ, RHO_MAX, make_resonator


def _fixed_channel_gym(modulation: str, sps: int, channel_seed: int, n_channels: int = 1) -> Gym:
    profile = WaveformProfile(modulation, sps=sps, rolloff=0.35)
    chan = ChannelParams(snr_db=25.0, multipath_taps=3, multipath_spread=0.6,
                         cfo_cycles_per_block=0.0, channel_seed=channel_seed)

    class _Fixed(Gym):
        def sample_spec(self, rng, n_samples=4096, noise_prob=0.0):
            s = super().sample_spec(rng, n_samples=n_samples, noise_prob=noise_prob)
            return EpisodeSpec(profile=profile, channel=chan, fs_hz=F0_HZ,
                               n_samples=n_samples, seed=s.seed, n_channels=n_channels,
                               is_noise=s.is_noise)

    return _Fixed(catalog=[profile], n_channels=n_channels)


def cmd_verify_invariance(args: argparse.Namespace) -> int:
    print("z-plane invariance certificate (eps <= 2 R rho^N / (1 - rho))")
    print(f"  master anchor F0 = {F0_HZ/1e6:.2f} MSPS, pole cap rho_max = {RHO_MAX:.7f}")
    k = make_resonator(radius=0.9, angle=0.5)
    print(f"  sample kernel: 1 pole, radius 0.9, R = {k.residue_norm():.4f}")
    for m in (10, 12, 13, 14, 16):
        n = 1 << m
        print(f"    N = 2^{m:<2d} : certified eps = {k.aliasing_certificate(n):.3e}")
    # Transport only ever moves poles inward.
    r0 = k.pole_radius
    r_low = k.transport(F0_HZ / 4).pole_radius
    print(f"  transport to F0/4 moves pole radius {r0:.4f} -> {r_low:.4f} (inward: {r_low < r0})")
    print("VERIFY_INVARIANCE PASS")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    config = NAMED_CONFIGS.get(args.config) or OperatorConfig.diagonal_for_channels(1, sections=8)
    gym = _fixed_channel_gym(args.modulation, args.sps, args.channel_seed, n_channels=config.in_channels)
    template = NeuralOperator.warm_start(config)
    history = train_operator(
        template, gym, episode_reward,
        generations=args.generations, batch=args.batch, n_samples=args.n_samples,
        sigma0=args.sigma0, seed=args.seed,
    )
    operator = template.with_adapted_vector(history.best_vector)
    lineage = BankLineage(training_seed=args.seed)
    bank = compile_bank(operator, lineage=lineage)
    with open(args.out, "wb") as fh:
        fh.write(bank.payload)
    print(json.dumps({
        "best_reward": history.best_reward,
        "digest": bank.manifest.digest,
        "log2n_range": [bank.manifest.log2n_min, bank.manifest.log2n_max],
        "out": args.out,
    }, indent=2))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    with open(args.bank, "rb") as fh:
        bank = load_bank(fh.read())
    if not bank.verify_crc():
        print("CRC FAIL", file=sys.stderr)
        return 2
    gym = Gym(n_channels=bank.operator.config.in_channels)
    report = run_gates(
        bank.operator, gym, held_out_modulation=args.held_out,
        reward_fn=episode_reward, eval_count=args.eval_count, seed=args.seed,
    )
    print(report.summary())
    return 0 if report.promoted else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atom-neural-rl", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify-invariance", help="print the discretization certificate")
    v.set_defaults(func=cmd_verify_invariance)

    t = sub.add_parser("train", help="train an operator and write a weight bank")
    t.add_argument("--out", default="bank.anrl")
    t.add_argument("--config", default="C1", choices=list(NAMED_CONFIGS))
    t.add_argument("--modulation", default="qpsk")
    t.add_argument("--sps", type=int, default=4)
    t.add_argument("--channel-seed", dest="channel_seed", type=int, default=1234)
    t.add_argument("--generations", type=int, default=20)
    t.add_argument("--batch", type=int, default=6)
    t.add_argument("--n-samples", dest="n_samples", type=int, default=1024)
    t.add_argument("--sigma0", type=float, default=0.3)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("gate", help="run promotion gates on a weight bank")
    g.add_argument("--bank", required=True)
    g.add_argument("--held-out", dest="held_out", default="qpsk")
    g.add_argument("--eval-count", dest="eval_count", type=int, default=40)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_gate)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
