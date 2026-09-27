from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

DELTA = 0.10
BUDGETS = [0.05, 0.10, 0.15, 0.20]
FINE = [round(0.01 * i, 2) for i in range(1, 31)]


def pvalues(k, n, alpha):
    # exact binomial p-value for H0: R > alpha, k = violations out of n
    return binom.cdf(np.asarray(k), n, alpha)


def holm(p, delta=DELTA):
    p = np.asarray(p, dtype=float)
    m = len(p)
    valid = np.zeros(m, dtype=bool)
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        if p[i] > delta / (m - rank):
            break
        valid[i] = True
    return valid


def deploy(valid, cost, ref, lower=False):
    # fastest valid config, first listed on ties, reference if nothing is valid
    best = None
    for i in np.flatnonzero(valid):
        if best is None or (cost[i] < cost[best] if lower else cost[i] > cost[best]):
            best = i
    return ref if best is None else int(best)


def run(grid, alpha, delta=DELTA, idx=None, cost=None):
    """Valid mask, deployed index and p-values of a Grid on prompts idx (all if None).

    Configs are ranked by cost, which defaults to the full-sample cost.
    """
    k = grid.violations(idx)
    n = grid.n if idx is None else len(idx)
    p = pvalues(k, n, alpha)
    valid = holm(p, delta)
    c = grid.cost() if cost is None else cost
    return valid, deploy(valid, c, grid.ref, grid.lower_is_better), p


def alpha_min(grid, delta=DELTA, alphas=FINE, idx=None, cost=None):
    # smallest budget on the grid at which something other than the reference is deployed
    for a in alphas:
        if run(grid, a, delta, idx, cost)[1] != grid.ref:
            return a
    return None


@dataclass
class Selection:
    """What select() returns. Arrays follow the order of the configurations."""

    names: list
    cost: np.ndarray
    accuracy: np.ndarray
    violations: np.ndarray
    n: int
    pvalues: np.ndarray
    valid: np.ndarray
    deployed: int
    reference: int
    alpha: float
    delta: float
    lower_is_better: bool = False

    @property
    def risk(self):
        # empirical reference-relative risk k/n
        return self.violations / self.n

    @property
    def deployed_name(self):
        return self.names[self.deployed]

    @property
    def gain(self):
        # percent gain of the deployed config over the reference, or the reduction factor for memory
        c, r = self.cost[self.deployed], self.cost[self.reference]
        return r / c if self.lower_is_better else 100.0 * (c / r - 1.0)

    def __str__(self):
        w = max(len(s) for s in self.names) + 6
        lines = [f"alpha={self.alpha:.2f}  delta={self.delta:.2f}  n={self.n}  m={len(self.names)}",
                 f"{'config':{w}s} {'cost':>6s} {'acc':>6s} {'risk':>6s} {'p-value':>8s}  valid"]
        for i, name in enumerate(self.names):
            tag = name + (" (ref)" if i == self.reference else "")
            mark = ("yes" + ("  <- deployed" if i == self.deployed else "")) if self.valid[i] else "no"
            lines.append(f"{tag:{w}s} {self.cost[i]:6.2f} {self.accuracy[i]:6.3f} {self.risk[i]:6.3f} "
                         f"{self.pvalues[i]:8.1e}  {mark}")
        if self.deployed == self.reference:
            lines.append(f"deploy {self.deployed_name}, nothing faster is valid")
        elif self.lower_is_better:
            lines.append(f"deploy {self.deployed_name}, {self.gain:.2f}x below the reference")
        else:
            lines.append(f"deploy {self.deployed_name}, {self.gain:+.1f}% over the reference")
        return "\n".join(lines)


def select(correct, cost, reference=0, alpha=0.10, delta=DELTA, names=None, lower_is_better=False):
    """Redline on a grid given as arrays.

    correct is an (m, n) 0/1 array, one row per configuration and one column per
    calibration prompt, and cost holds one value per configuration (TPF, or
    memory with lower_is_better). Deploys the fastest configuration that passes
    the exact binomial test under Holm at level delta.
    """
    correct = np.asarray(correct, dtype=bool)
    cost = np.asarray(cost, dtype=float)
    m, n = correct.shape
    if cost.shape != (m,):
        raise ValueError(f"cost needs one value per configuration, got shape {cost.shape} for m={m}")
    names = [str(i) for i in range(m)] if names is None else [str(s) for s in names]
    k = (correct[reference][None, :] & ~correct).sum(1)
    p = pvalues(k, n, alpha)
    valid = holm(p, delta)
    i = deploy(valid, cost, reference, lower_is_better)
    return Selection(names, cost, correct.mean(1), k, n, p, valid, i, reference, alpha, delta,
                     lower_is_better)
