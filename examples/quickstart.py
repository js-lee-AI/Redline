"""Redline on a toy grid. CPU only, no downloads, well under a second."""

import numpy as np

import redline

rng = np.random.default_rng(0)
ref = rng.random(1000) < 0.8                    # the reference answers 80% of 1,000 prompts
risk = np.array([0.03, 0.05, 0.07, 0.13])       # true P(reference right, config wrong)
fix = np.array([0.05, 0.05, 0.10, 0.50])        # share of reference misses each config gets right
u = rng.random((4, 1000))
faster = np.where(ref, u >= risk[:, None] / 0.8, u < fix[:, None])
correct = np.vstack([ref, faster])              # one row of 0/1 answers per configuration
tpf = [4.0, 4.6, 5.2, 5.8, 6.4]                 # tokens per forward, reference first

print(redline.select(correct, tpf, reference=0, alpha=0.10, names=["default", "A", "B", "C", "D"]))
