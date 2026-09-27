"""Full grid: cost, forwards and tokens per request, joint risk and p-values at four budgets.

Table 4 is sdar_math500_distilled, Tables 8 to 11 are llada2_math, llada2_code, sdar_math and sdar_code,
Tables 12 and 13 are llada2_math_small and llada2_code_small, and llada2_math_3d is the grid of Appendix D.7.
* = valid under Holm, < = deployed.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from redline import BUDGETS, DATA_DIR, DELTA, load_grid, run  # noqa: E402


def fmt_p(p):
    if p >= 1e-3:
        return f"{p:.3f}"
    m, e = f"{p:.0e}".split("e")
    return f"{m}e{int(e)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--delta", type=float, default=DELTA)
    args = ap.parse_args()
    g = load_grid(args.grid, args.data_dir)
    cost = g.cost()
    k = g.violations()
    runs = {a: run(g, a, args.delta) for a in BUDGETS}

    unit = {"tpf": "TPF", "apf": "APF", "memory": "GB"}[g.kind]
    print(f"{g.label}  n={g.n}  m={g.m}  delta={args.delta}")
    print(f"{'configuration':26s} {unit:>7s} {'fwd/req':>8s} {'tok/req':>8s} {'R_hat (k/n)':>18s}"
          + "".join(f" {'a=%.2f' % a:>9s}" for a in BUDGETS))
    # hand-picked before distilled, then by speed
    order = sorted(range(g.m), key=lambda i: (g.configs[i].startswith("distilled/"),
                                              cost[i] if g.lower_is_better else -cost[i]))
    for i in order:
        name = g.configs[i] + (" (ref)" if i == g.ref else "")
        fwd = "-" if g.forwards is None else f"{g.forwards[i].mean():.1f}"
        line = (f"{name:26s} {cost[i]:7.3f} {fwd:>8s} {g.tokens[i].mean():8.0f} "
                f"{f'{k[i] / g.n:.3f} ({k[i]}/{g.n})':>18s}")
        for a in BUDGETS:
            valid, dep, p = runs[a]
            mark = ("*" if valid[i] else "") + ("<" if dep == i else "")
            line += f" {fmt_p(p[i]) + mark:>9s}"
        print(line)
    ref_ok = g.correct[g.ref]
    print(f"reference accuracy {100 * ref_ok.mean():.1f}%, "
          f"share of reference-correct answers lost: " +
          ", ".join(f"{g.configs[i]} {100 * k[i] / ref_ok.sum():.1f}%" for i in order if i != g.ref))
    for a in BUDGETS:
        valid, dep, _ = runs[a]
        extra = "" if g.kind == "memory" else f" gain {g.gain(dep):+.1f}%"
        print(f"a={a:.2f}: {int(valid.sum())} of {g.m} valid, deployed {g.configs[dep]}{extra} "
              f"net {g.net(dep):+.1f}pp")
    if g.forwards is not None:
        f = g.forwards.mean(1)
        t = g.tokens.mean(1)
        dep = runs[0.10][1]
        if dep != g.ref:
            print(f"a=0.10 deployed vs reference: forwards per request {100 * (f[dep] / f[g.ref] - 1):+.1f}%, "
                  f"tokens per request {100 * (t[dep] / t[g.ref] - 1):+.1f}%")


if __name__ == "__main__":
    main()
