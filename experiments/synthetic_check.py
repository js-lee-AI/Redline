"""Synthetic check with known risks (Appendix A.7): Holm on five dependent configurations."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from redline import holm, pvalues  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()
    risks = np.array([0.01, 0.02, 0.07, 0.07, 0.07])  # two below the budget, three above
    rng = np.random.default_rng(args.seed)
    bad = 0
    for _ in range(args.trials):
        u = rng.random(args.n)  # shared per prompt, so the losses are positively dependent
        k = (u[None, :] < risks[:, None]).sum(1)
        valid = holm(pvalues(k, args.n, args.alpha), args.delta)
        bad += bool((risks[valid] > args.alpha).any())
    print(f"n={args.n} trials={args.trials} alpha={args.alpha} delta={args.delta}: "
          f"family-wise error rate {bad / args.trials:.4f}")


if __name__ == "__main__":
    main()
