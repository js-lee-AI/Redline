"""Held-out re-measurement (Appendix D.10, Table 16): Redline on one random half, measured on the other.

Also prints, per budget, the mean deployed gain over splits and the held-out exceedance (the Redline point
of Figure 3b and the Redline column of Tables 14 and 15), and the exceedance against the risk over all
prompts. Run with --grids llada2_math_large for the held-out numbers of the 50-configuration grid.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from redline import BUDGETS, DATA_DIR, DELTA, deploy, holm, load_grid, pvalues, run  # noqa: E402

GRIDS = ["llada2_math", "llada2_code", "sdar_math", "sdar_code"]
SEED = 20260918


def half_splits(n, K, seed=SEED, frac=0.5):
    rng = np.random.default_rng(seed)
    n_cal = int(round(frac * n))
    out = []
    for _ in range(K):
        perm = rng.permutation(n)
        out.append((perm[:n_cal], perm[n_cal:]))
    return out


def pm(x):
    return f"{x:+.1f}"


def measure(g, K, delta=DELTA):
    cost = g.cost()
    rows = {}
    rec = {a: dict(dep=[], gain=[], net=[], risk=[], full_gain=[]) for a in BUDGETS}
    for cal, test in half_splits(g.n, K):
        k = g.joint[:, cal].sum(1)
        c_test = g.cost(test)
        acc_test = g.accuracy(test)
        r_test = g.joint[:, test].mean(1)
        for a in BUDGETS:
            valid = holm(pvalues(k, len(cal), a), delta)
            i = deploy(valid, cost, g.ref, g.lower_is_better)
            r = rec[a]
            r["dep"].append(i)
            r["gain"].append(100.0 * (c_test[i] / c_test[g.ref] - 1.0))
            r["net"].append(100.0 * (acc_test[i] - acc_test[g.ref]))
            r["risk"].append(r_test[i])
            r["full_gain"].append(g.gain(i) if i != g.ref else 0.0)
    pooled = g.joint.mean(1)
    for a in BUDGETS:
        r = rec[a]
        f = run(g, a, delta)[1]
        dep = np.array(r["dep"])
        same = dep == f
        row = dict(deployed=g.configs[f], is_ref=f == g.ref, in_gain=g.gain(f), in_net=g.net(f),
                   n_ref=int((dep == g.ref).sum()), n_same=int(same.sum()),
                   exceed=float(np.mean(np.array(r["risk"]) > a)), exceed_pooled=float(np.mean(pooled[dep] > a)),
                   mean_gain=float(np.mean(r["full_gain"])))
        if f != g.ref:
            sg, sn = np.array(r["gain"])[same], np.array(r["net"])[same]
            row.update(gain=float(sg.mean()), gain_lo=float(np.percentile(sg, 2.5)), gain_hi=float(np.percentile(sg, 97.5)),
                       net=float(sn.mean()), net_lo=float(np.percentile(sn, 2.5)), net_hi=float(np.percentile(sn, 97.5)))
        rows[a] = row
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--grids", default=",".join(GRIDS))
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--delta", type=float, default=DELTA)
    args = ap.parse_args()

    for name in args.grids.split(","):
        g = load_grid(name, args.data_dir)
        rows = measure(g, args.K, args.delta)
        print(f"{g.label} (n={g.n}, m={g.m}, K={args.K} half splits)")
        print(f"  {'alpha':>5s} {'deployed':22s} {'in-sample':>12s} {'ref':>5s} {'same':>5s} "
              f"{'held-out gain [2.5, 97.5]':>26s} {'held-out net [2.5, 97.5]':>26s} "
              f"{'exceed':>7s} {'pooled':>7s} {'mean gain':>9s}")
        for a in BUDGETS:
            r = rows[a]
            if r["is_ref"]:
                ins, held_g, held_n = "+0.0 0.0", "+0.0", "0.0"
            else:
                ins = f"{pm(r['in_gain'])} {pm(r['in_net'])}"
                held_g = f"{pm(r['gain'])} [{pm(r['gain_lo'])}, {pm(r['gain_hi'])}]"
                held_n = f"{pm(r['net'])} [{pm(r['net_lo'])}, {pm(r['net_hi'])}]"
            print(f"  {a:5.2f} {r['deployed']:22s} {ins:>12s} {r['n_ref']:5d} {r['n_same']:5d} "
                  f"{held_g:>26s} {held_n:>26s} {100 * r['exceed']:6.1f}% {100 * r['exceed_pooled']:6.1f}% "
                  f"{r['mean_gain']:+8.1f}%")


if __name__ == "__main__":
    main()
