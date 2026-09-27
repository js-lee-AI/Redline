"""Calibration size for the exact binomial test (paper Table 7). No data needed."""

import numpy as np
from scipy.stats import binom

N_MAX = 60000


def critical_count(n, alpha, level):
    # largest violation count s with P(Bin(n, alpha) <= s) <= level, -1 if none
    s = int(binom.ppf(level, n, alpha))
    while s >= 0 and binom.cdf(s, n, alpha) > level:
        s -= 1
    while binom.cdf(s + 1, n, alpha) <= level:
        s += 1
    return s


def prompts_needed(alpha, risk, level, power=0.8, n_max=N_MAX):
    """Prompts from which a config of true risk `risk` passes a test at `level` with
    probability >= power, for every larger n up to n_max. None if not reached by n_max.

    Use level = delta / m for Holm's first step on a grid of m configurations.
    """
    ns = np.arange(20, n_max + 1)
    s = binom.ppf(level, ns, alpha)
    c = np.where(binom.cdf(s, ns, alpha) <= level, s, s - 1)
    ok = (c >= 0) & (binom.cdf(c, ns, risk) >= power)
    if not ok[-1]:
        return None
    bad = np.flatnonzero(~ok)
    # power is saw-toothed in n, so report where it stays above the target
    return int(ns[bad[-1] + 1]) if len(bad) else int(ns[0])
