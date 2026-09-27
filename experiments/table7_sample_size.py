"""Table 7: calibration size for the exact binomial test (no data needed).

(a) the n from which a config of true risk R passes with probability >= 0.8 at Holm's first step
delta/m, for every larger n up to the scan limit; (b) the largest empirical risk that passes at alpha = 0.10.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from redline import critical_count, prompts_needed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.10)
    args = ap.parse_args()
    d = args.delta
    print(f"(a) prompts for power 0.8 at delta/m, delta = {d}")
    print(f"  {'alpha':>5s} {'true R':>6s} {'m=8':>7s} {'m=16':>7s} {'m=50':>7s}")
    for alpha, r in [(0.02, 0.01), (0.05, 0.04), (0.05, 0.03), (0.10, 0.09), (0.10, 0.08)]:
        print(f"  {alpha:5.2f} {r:6.2f}" + "".join(f" {prompts_needed(alpha, r, d / m):7d}" for m in (8, 16, 50)))
    print("(b) largest R_hat (%) that passes at alpha = 0.10")
    print(f"  {'n':>5s} {'delta/8':>8s} {'delta/50':>8s} {'delta':>8s}")
    for n in (542, 664, 1012, 2000, 5000):
        print(f"  {n:5d}" + "".join(f" {100 * (critical_count(n, 0.10, t) / n):8.1f}" for t in (d / 8, d / 50, d)))


if __name__ == "__main__":
    main()
