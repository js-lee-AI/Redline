"""Table 17: the 50-configuration LLaDA2 math grid as three nested grids (m = 6, 25, 50)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from redline import BUDGETS, DATA_DIR, DELTA, alpha_min, load_grid, run  # noqa: E402


def nested(g):
    six = [c for c in g.configs if c.endswith("add0.10") and c.split("/")[0] in ("acc85", "acc90", "acc95")
           and c.split("/")[1] in ("semi70", "semi90")]
    add10 = [c for c in g.configs if c.endswith("add0.10")]
    return [("shared with the main grid", six), ("add 0.10 slice", add10), ("whole grid", g.configs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--delta", type=float, default=DELTA)
    args = ap.parse_args()
    full = load_grid("llada2_math_large", args.data_dir)
    k = full.violations()
    for title, names in nested(full):
        g = full.subset(names)
        print(f"{title}, m = {g.m}, n = {g.n}, reference {g.configs[g.ref]}, "
              f"first gain at alpha = {alpha_min(g, args.delta)}")
        print(f"  {'alpha':>5s} {'deployed':22s} {'valid':>9s} {'k':>4s} {'R_hat':>6s} {'TPF':>5s} "
              f"{'gain':>6s} {'net':>5s}")
        cost = g.cost()
        for a in BUDGETS:
            valid, i, _ = run(g, a, args.delta)
            ki = int(k[full.configs.index(g.configs[i])])
            print(f"  {a:5.2f} {g.configs[i]:22s} {f'{int(valid.sum())} of {g.m}':>9s} {ki:4d} {ki / g.n:6.3f} "
                  f"{cost[i]:5.2f} {g.gain(i):+6.1f} {g.net(i):+5.1f}")


if __name__ == "__main__":
    main()
