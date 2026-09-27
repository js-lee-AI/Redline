"""Table 6: smallest math budget with a gain when the math set is subsampled to the code n."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from redline import DATA_DIR, DELTA, alpha_min, load_grid  # noqa: E402

FAMILIES = [("LLaDA2", "llada2_math", "llada2_code"), ("SDAR", "sdar_math", "sdar_code")]
SEEDS = {"uniform": 20260919, "stratified": 20260929}


def draws(g, n_code, K, arm):
    rng = np.random.default_rng(SEEDS[arm])
    gsm = np.array([i for i, p in enumerate(g.prompts) if p.startswith("gsm8k/")])
    m500 = np.array([i for i, p in enumerate(g.prompts) if p.startswith("math500/")])
    n_gsm = int(round(n_code * len(gsm) / g.n))  # keep the GSM8K share of the suite
    for _ in range(K):
        if arm == "uniform":
            yield np.sort(rng.choice(g.n, size=n_code, replace=False))
        else:
            yield np.sort(np.concatenate([rng.choice(gsm, size=n_gsm, replace=False),
                                          rng.choice(m500, size=n_code - n_gsm, replace=False)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--delta", type=float, default=DELTA)
    args = ap.parse_args()
    for fam, math_name, code_name in FAMILIES:
        g = load_grid(math_name, args.data_dir)
        code = load_grid(code_name, args.data_dir)
        cost = g.cost()
        print(f"{fam}: math n={g.n} first gain at {alpha_min(g, args.delta):.2f}, "
              f"code n={code.n} first gain at {alpha_min(code, args.delta):.2f}")
        for arm in SEEDS:
            fu = []
            for idx in draws(g, code.n, args.K, arm):
                a = alpha_min(g, args.delta, idx=idx, cost=cost)
                fu.append(np.inf if a is None else a)
            fu = np.array(fu)
            hist = [int(np.isclose(fu, 0.01 * i).sum()) for i in range(5, 13)]
            print(f"  {arm:10s} n={code.n} median {np.median(fu):.2f} IQR [{np.percentile(fu, 25):.2f}, "
                  f"{np.percentile(fu, 75):.2f}]  <=0.10 {np.mean(fu <= 0.10 + 1e-9):.3f}  "
                  f"<0.11 {np.mean(fu < 0.11 - 1e-9):.3f}  <=0.11 {np.mean(fu <= 0.11 + 1e-9):.3f}  "
                  f"draws at 0.05..0.12: {', '.join(map(str, hist))}")


if __name__ == "__main__":
    main()
