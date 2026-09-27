import copy
import csv
import json
import os

import numpy as np

# data/ of a git checkout; a pip install has no data, so pass data_dir or use load_outcomes
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class Grid:
    def __init__(self, name, spec, configs, prompts, correct, tokens, forwards):
        self.name = name
        self.label = spec.get("label", name)
        self.kind = spec["cost"]  # tpf, apf or memory
        self.configs = configs
        self.prompts = prompts
        self.correct = correct
        self.tokens = tokens
        self.forwards = forwards
        if spec["reference"] not in configs:
            raise ValueError(f"reference {spec['reference']!r} is not one of the configs {configs}")
        self.ref = configs.index(spec["reference"])
        self.memory = np.array([spec["memory_gb"][c] for c in configs]) if self.kind == "memory" else None
        self.lower_is_better = self.kind == "memory"
        # joint loss: reference right and config wrong
        self.joint = self.correct[self.ref][None, :] & ~self.correct

    @property
    def m(self):
        return len(self.configs)

    @property
    def n(self):
        return len(self.prompts)

    def violations(self, idx=None):
        j = self.joint if idx is None else self.joint[:, idx]
        return j.sum(1)

    def accuracy(self, idx=None):
        c = self.correct if idx is None else self.correct[:, idx]
        return c.mean(1)

    def cost(self, idx=None):
        if self.kind == "memory":
            return self.memory.copy()
        tok = self.tokens if idx is None else self.tokens[:, idx]
        fwd = self.forwards if idx is None else self.forwards[:, idx]
        if self.kind == "tpf":
            return tok.sum(1) / fwd.sum(1)
        return (tok / np.maximum(fwd, 1)).mean(1)

    def gain(self, i, idx=None):
        # percent gain over the reference, or the memory reduction factor
        c = self.cost(idx)
        if self.lower_is_better:
            return c[self.ref] / c[i]
        return 100.0 * (c[i] / c[self.ref] - 1.0)

    def net(self, i, idx=None):
        a = self.accuracy(idx)
        return 100.0 * (a[i] - a[self.ref])

    def subset(self, names):
        # same outcomes, fewer configs (must keep the reference)
        idx = [self.configs.index(c) for c in names]
        g = copy.copy(self)
        g.configs = list(names)
        g.correct, g.tokens, g.joint = self.correct[idx], self.tokens[idx], self.joint[idx]
        g.forwards = None if self.forwards is None else self.forwards[idx]
        g.memory = None if self.memory is None else self.memory[idx]
        g.ref = g.configs.index(self.configs[self.ref])
        return g


def grid_specs(data_dir=DATA_DIR):
    path = os.path.join(data_dir, "grids.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. The shipped grids come with a git clone of the repository.")
    with open(path) as f:
        return json.load(f)


def read_outcomes(path):
    rows = {}
    prompts = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["config"], []).append(r)
            prompts.setdefault(r["config"], []).append(r["prompt"])
    order = list(rows)
    for c in order:
        assert prompts[c] == prompts[order[0]], f"{path}: prompt order differs for {c}"
    return order, prompts[order[0]], rows


def _build(name, spec, path):
    order, prompts, rows = read_outcomes(path)
    configs = spec.get("configs", order)
    keep = np.arange(len(prompts))
    if "prompts" in spec:
        keep = np.array([i for i, p in enumerate(prompts) if p.split("/")[0] == spec["prompts"]])
    correct = np.array([[int(rows[c][i]["correct"]) for i in keep] for c in configs], dtype=bool)
    tokens = np.array([[float(rows[c][i]["tokens"]) for i in keep] for c in configs])
    forwards = None
    if spec["cost"] != "memory":
        forwards = np.array([[float(rows[c][i]["forwards"]) for i in keep] for c in configs])
    return Grid(name, spec, list(configs), [prompts[i] for i in keep], correct, tokens, forwards)


def load_grid(name, data_dir=DATA_DIR):
    # one of the grids of the paper, as defined in data/grids.json
    spec = grid_specs(data_dir)[name]
    return _build(name, spec, os.path.join(data_dir, "outcomes", spec["file"]))


def load_outcomes(path, reference, cost="tpf", memory_gb=None):
    """A Grid from your own outcome file with columns config, prompt, correct, tokens, forwards.

    cost is tpf (tokens over forwards, pooled), apf (per-prompt ratio, averaged)
    or memory, which ranks by memory_gb, a dict from config to gigabytes.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    spec = {"cost": cost, "reference": reference, "label": name}
    if cost == "memory":
        spec["memory_gb"] = memory_gb
    return _build(name, spec, path)
