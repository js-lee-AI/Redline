"""Table 5: intervals for every deployed configuration other than the reference.

Paired bootstrap over prompts (the same resample weights the deployed config and the reference), and the
Clopper-Pearson interval and one-sided 90% upper bound of the joint risk.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from redline import BUDGETS, DATA_DIR, DELTA, load_grid, run  # noqa: E402

GRIDS = ["llada2_math", "llada2_math_large", "llada2_code", "sdar_math", "sdar_code",
         "specdec_llama_gsm8k", "specdec_qwen_gsm8k", "quant_llama_gsm8k"]
SEED = 20260920


def boot_weights(n, B):
    rng = np.random.default_rng(SEED)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)


def ci(x):
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--grids", default=",".join(GRIDS))
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--delta", type=float, default=DELTA)
    args = ap.parse_args()
    with open(os.path.join(args.data_dir, "quant_memory.json")) as f:
        weight_gb = json.load(f)["weight_gb"]
    rows = []
    for name in args.grids.split(","):
        g = load_grid(name, args.data_dir)
        n, ref = g.n, g.ref
        W = boot_weights(n, args.B)
        ok = g.correct.astype(float)
        bn_ref = W @ ok[ref] / n
        deployed = {}
        for a in BUDGETS:
            i = run(g, a, args.delta)[1]
            if i != ref:
                deployed.setdefault(i, []).append(a)
        for i, alphas in deployed.items():
            if g.kind == "tpf":
                bg = 100.0 * ((W @ g.tokens[i]) / (W @ g.forwards[i]) / ((W @ g.tokens[ref]) / (W @ g.forwards[ref])) - 1.0)
            elif g.kind == "apf":
                c = g.tokens / np.maximum(g.forwards, 1)
                bg = 100.0 * ((W @ c[i]) / (W @ c[ref]) - 1.0)
            else:
                bg = None
            bn = 100.0 * (W @ ok[i] / n - bn_ref)
            k = int(g.violations()[i])
            lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
            hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
            ub = beta.ppf(1 - args.delta, k + 1, n - k)
            if bg is None:
                gain = f"{weight_gb[g.configs[ref]] / weight_gb[g.configs[i]]:.2f}x measured"
            else:
                gain = f"{g.gain(i):+.1f} [{ci(bg)[0]:.1f}, {ci(bg)[1]:.1f}]"
            rows.append(dict(bg=bg, ub=ub, alphas=alphas))
            print(f"{g.label:34s} a={','.join(f'{a:.2f}' for a in alphas):14s} {g.configs[i]:22s} gain {gain:22s} "
                  f"net {g.net(i):+.1f} [{ci(bn)[0]:+.1f}, {ci(bn)[1]:+.1f}]  "
                  f"R_hat {k / n:.3f} [{lo:.3f}, {hi:.3f}] upper {ub:.3f}")
    entries = sum(len(r["alphas"]) for r in rows)
    speed = [r for r in rows if r["bg"] is not None]
    print(f"\n{entries} deployments at the budgets {BUDGETS}: "
          f"{sum(len(r['alphas']) for r in speed if ci(r['bg'])[0] > 0)} of "
          f"{sum(len(r['alphas']) for r in speed)} speed intervals above zero, "
          f"{sum(sum(r['ub'] < a for a in r['alphas']) for r in rows)} of {entries} upper bounds below the budget")


if __name__ == "__main__":
    main()
