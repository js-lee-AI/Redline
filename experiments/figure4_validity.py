"""Out-of-sample validity (Section 5.5, Figure 4): random calibration/test splits of 12 grids.

For every grid, budget and calibration size, Redline runs on the calibration prompts and we record
coverage (deployed config has test risk <= alpha), the family-wise error rate (some valid config has
test risk > alpha) and the same error rate against the risk over all n prompts (pooled).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from redline import BUDGETS, DATA_DIR, DELTA, deploy, holm, load_grid, pvalues  # noqa: E402

GRIDS = ["llada2_math", "llada2_code", "sdar_math", "sdar_code", "llada2_math_small", "llada2_code_small",
         "llada2_math_tiny", "specdec_llama_gsm8k_all", "specdec_qwen_gsm8k_all", "specdec_llama_humaneval",
         "specdec_qwen_humaneval", "quant_llama_gsm8k"]


def cal_sizes(n):
    out = sorted({c for c in (n // 2, 200, 100, 50, 25) if 15 <= c <= n - 15})
    return out or [max(10, n // 2)]


def run_grid(g, K, delta):
    n = g.n
    joint = g.joint
    cost = g.cost()
    pooled = joint.mean(1)
    sizes = cal_sizes(n)
    cov = {(nc, a): 0 for nc in sizes for a in BUDGETS}
    fwer = dict.fromkeys(cov, 0)
    fwer_pool = dict.fromkeys(cov, 0)
    for s in range(K):
        perm = np.random.default_rng(s).permutation(n)  # seed = split index
        for nc in sizes:
            cal, test = perm[:nc], perm[nc:]
            k = joint[:, cal].sum(1)
            test_r = joint[:, test].mean(1)
            for a in BUDGETS:
                valid = holm(pvalues(k, nc, a), delta)
                dep = deploy(valid, cost, g.ref, g.lower_is_better)
                cov[(nc, a)] += test_r[dep] <= a
                fwer[(nc, a)] += bool((test_r[valid] > a).any())
                fwer_pool[(nc, a)] += bool((pooled[valid] > a).any())
    return [dict(alpha=a, n_cal=nc, coverage=cov[(nc, a)] / K, fwer=fwer[(nc, a)] / K,
                 fwer_pooled=fwer_pool[(nc, a)] / K) for nc in sizes for a in BUDGETS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--grids", default=",".join(GRIDS))
    ap.add_argument("--K", type=int, default=1000, help="random splits per grid")
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--out", default=None, help="optional json with every combination")
    args = ap.parse_args()
    t0 = time.time()
    allrows = {}
    for name in args.grids.split(","):
        g = load_grid(name, args.data_dir)
        rows = run_grid(g, args.K, args.delta)
        allrows[name] = rows
        print(f"{g.label} (n={g.n}, m={g.m}, calibration sizes {cal_sizes(g.n)})")
        for a in BUDGETS:
            r = [x for x in rows if x["alpha"] == a]
            print(f"  alpha={a:.2f}  min coverage {min(x['coverage'] for x in r):.3f}  "
                  f"max fwer {max(x['fwer'] for x in r):.3f}  max pooled fwer {max(x['fwer_pooled'] for x in r):.3f}")
    flat = [x for rows in allrows.values() for x in rows]
    cmin = min(x["coverage"] for x in flat)
    fmax = max(x["fwer"] for x in flat)
    se = np.sqrt(fmax * (1 - fmax) / args.K)
    print(f"\n{len(allrows)} grids, {len(flat)} combinations, K = {args.K} splits each ({time.time() - t0:.0f}s)")
    print(f"min coverage {cmin:.3f}, max fwer {fmax:.3f} (MC s.e. {se:.3f}), "
          f"max pooled fwer {max(x['fwer_pooled'] for x in flat):.3f}")
    print(f"combinations with fwer > delta: {sum(x['fwer'] > args.delta for x in flat)} (test half), "
          f"{sum(x['fwer_pooled'] > args.delta for x in flat)} (pooled)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(K=args.K, delta=args.delta, grids=allrows), f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
