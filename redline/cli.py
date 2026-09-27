"""The `redline` command. `python -m redline` runs the same thing."""

from __future__ import annotations

import argparse
import json

from . import __version__
from .core import DELTA


def _demo(args):
    import numpy as np

    from . import select

    # the grid of examples/quickstart.py
    rng = np.random.default_rng(args.seed)
    ref = rng.random(1000) < 0.8
    risk = np.array([0.03, 0.05, 0.07, 0.13])
    fix = np.array([0.05, 0.05, 0.10, 0.50])
    u = rng.random((4, 1000))
    faster = np.where(ref, u >= risk[:, None] / 0.8, u < fix[:, None])
    correct = np.vstack([ref, faster])
    tpf = [4.0, 4.6, 5.2, 5.8, 6.4]
    print(select(correct, tpf, reference=0, alpha=args.alpha, delta=args.delta,
                 names=["default", "A", "B", "C", "D"]))
    return 0


def _select(args):
    from . import load_outcomes, select

    g = load_outcomes(args.outcomes, args.reference, cost=args.cost)
    res = select(g.correct, g.cost(), reference=g.ref, alpha=args.alpha, delta=args.delta, names=g.configs)
    print(res)
    if args.json:
        out = dict(alpha=args.alpha, delta=args.delta, n=res.n, reference=res.names[res.reference],
                   deployed=res.deployed_name, gain=float(res.gain),
                   configs=[dict(name=s, cost=float(res.cost[i]), accuracy=float(res.accuracy[i]),
                                 violations=int(res.violations[i]), risk=float(res.risk[i]),
                                 pvalue=float(res.pvalues[i]), valid=bool(res.valid[i]))
                            for i, s in enumerate(res.names)])
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", args.json)
    return 0


def _sample_size(args):
    from . import critical_count, prompts_needed

    level = args.delta / args.m
    n = prompts_needed(args.alpha, args.risk, level, args.power)
    if n is None:
        print(f"more than 60000 prompts are needed at alpha = {args.alpha} and true risk {args.risk}")
    else:
        print(f"{n} prompts let a configuration of true risk {args.risk} pass alpha = {args.alpha} with "
              f"probability {args.power} at Holm's first step (delta/m = {level:.4g}), and at every larger n")
    if args.n:
        s = critical_count(args.n, args.alpha, level)
        print(f"with {args.n} prompts it passes that step when its empirical risk is at most "
              f"{100 * s / args.n:.1f}% ({s} of {args.n})")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="redline", description="Deploy the fastest serving configuration within a risk budget")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the CPU quickstart on a toy grid")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--alpha", type=float, default=0.10, help="risk budget")
    demo.add_argument("--delta", type=float, default=DELTA, help="failure probability")
    demo.set_defaults(func=_demo)

    sel = sub.add_parser("select", help="run Redline on an outcome file of your own grid")
    sel.add_argument("outcomes", help="csv with columns config, prompt, correct, tokens, forwards")
    sel.add_argument("--reference", required=True, help="name of the reference configuration")
    sel.add_argument("--alpha", type=float, default=0.10, help="risk budget")
    sel.add_argument("--delta", type=float, default=DELTA, help="failure probability")
    sel.add_argument("--cost", choices=["tpf", "apf"], default="tpf",
                     help="tpf pools tokens over forwards, apf averages the per-prompt ratio")
    sel.add_argument("--json", default=None, help="also write the result to this file")
    sel.set_defaults(func=_select)

    size = sub.add_parser("sample-size", help="calibration prompts needed to pass a budget (paper Table 7)")
    size.add_argument("--alpha", type=float, required=True, help="risk budget")
    size.add_argument("--risk", type=float, required=True, help="true risk of the configuration")
    size.add_argument("--m", type=int, default=8, help="configurations in the grid")
    size.add_argument("--delta", type=float, default=DELTA, help="failure probability")
    size.add_argument("--power", type=float, default=0.8)
    size.add_argument("--n", type=int, default=None, help="also print the passing margin at this n")
    size.set_defaults(func=_sample_size)

    args = parser.parse_args(argv)
    return args.func(args)
