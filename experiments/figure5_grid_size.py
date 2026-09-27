"""Figure 5: deployed gain against grid size, random sub-grids of the 50-configuration LLaDA2 math grid.

Each sub-grid holds the reference and m - 1 other configurations drawn without replacement, and Redline
runs on all n prompts.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from redline import BUDGETS, DATA_DIR, DELTA, deploy, holm, load_grid, pvalues  # noqa: E402
from table17_large_grid import nested  # noqa: E402

SEED = 20260921


def subgrid_gains(g, sizes, K, delta=DELTA, seed=SEED):
    k, cost, n = g.violations(), g.cost(), g.n
    gain = 100.0 * (cost / cost[g.ref] - 1.0)
    others = [i for i in range(g.m) if i != g.ref]
    rng = np.random.default_rng(seed)

    def run(idx, a):
        idx = np.asarray(idx)
        valid = holm(pvalues(k[idx], n, a), delta)
        return idx[deploy(valid, cost[idx], 0)]

    res = {}
    for m in sizes:
        draws = [[g.ref] + [others[j] for j in rng.choice(len(others), size=m - 1, replace=False)]
                 for _ in range(K)]
        for a in BUDGETS:
            d = np.array([run(s, a) for s in draws])
            x = gain[d]
            res[(m, a)] = (x.mean(), x.std(ddof=1) / np.sqrt(len(x)), np.mean(d == g.ref))
    for a in BUDGETS:
        i = run([g.ref] + others, a)
        res[(g.m, a)] = (gain[i], 0.0, float(i == g.ref))
    return res, run, gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--sizes", default="8,16,32")
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--plot", default=None, help="optional output figure, e.g. gridsize.pdf")
    args = ap.parse_args()
    g = load_grid("llada2_math_large", args.data_dir)
    res, run, gain = subgrid_gains(g, [int(x) for x in args.sizes.split(",")], args.K, args.delta)

    sizes = sorted({m for m, _ in res})
    print(f"mean deployed TPF gain (%) over {args.K} random sub-grids, with Monte Carlo s.e. "
          f"and the share deploying the reference (m = {g.m} is the whole grid)")
    for a in BUDGETS:
        print(f"  alpha={a:.2f}  " + "  ".join(
            f"m={m}: {res[(m, a)][0]:+.1f} (s.e. {res[(m, a)][1]:.2f}, ref {res[(m, a)][2]:.2f})" for m in sizes))
    print("designed nested grids:")
    for title, names in nested(g):
        sub = [g.configs.index(c) for c in names]
        sub = [g.ref] + [i for i in sub if i != g.ref]
        print(f"  {title} (m={len(sub)}): " + "  ".join(f"alpha={a:.2f} {gain[run(sub, a)]:+.1f}" for a in BUDGETS))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for a in BUDGETS[1:]:
            ax.plot(sizes, [res[(m, a)][0] for m in sizes], marker="o", label=f"alpha = {a:.2f}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(m) for m in sizes])
        ax.set_xlabel("grid size m")
        ax.set_ylabel("deployed TPF gain (%)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
