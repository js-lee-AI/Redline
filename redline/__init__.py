"""Redline: faster block-diffusion serving with distribution-free risk guarantees.

    import redline
    result = redline.select(correct, tpf, reference=0, alpha=0.10)

correct holds one row of 0/1 answers per configuration on the same calibration
prompts. The deployed configuration has reference-relative risk at most alpha
with probability at least 1 - delta. Everything runs on numpy and scipy.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core import BUDGETS, DELTA, FINE, Selection, alpha_min, deploy, holm, pvalues, run, select
from .grids import DATA_DIR, Grid, grid_specs, load_grid, load_outcomes
from .power import critical_count, prompts_needed

__all__ = [
    "__version__", "select", "Selection", "load_outcomes", "prompts_needed", "critical_count",
    "pvalues", "holm", "deploy", "run", "alpha_min", "Grid", "load_grid", "grid_specs",
    "DATA_DIR", "DELTA", "BUDGETS", "FINE",
]
