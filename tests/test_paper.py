"""Paper numbers recomputed from the outcome files in data/ (a git checkout has them)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import redline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
pytestmark = pytest.mark.skipif(not (DATA / "grids.json").exists(), reason="needs the data/ folder of a clone")

# Table 2, grid: (n, m, alpha_min, [(gain, net accuracy change) at alpha = 0.10, 0.15, 0.20])
TABLE2 = {
    "llada2_math": (1012, 7, 0.07, [(37.5, -4.0), (37.5, -4.0), (37.5, -4.0)]),
    "llada2_math_large": (1012, 50, 0.07, [(65.4, -3.4), (72.0, -5.9), (72.0, -5.9)]),
    "llada2_code": (542, 8, 0.11, [(0.0, 0.0), (33.7, -7.0), (42.4, -13.1)]),
    "sdar_math": (1012, 8, 0.10, [(13.1, -2.5), (36.6, -4.6), (36.6, -4.6)]),
    "sdar_code": (664, 8, 0.11, [(0.0, 0.0), (25.7, -5.0), (59.7, -14.0)]),
    "specdec_llama_gsm8k": (512, 7, 0.09, [(13.8, -3.3), (16.8, -9.0), (16.8, -9.0)]),
    "specdec_qwen_gsm8k": (512, 7, 0.04, [(18.1, -1.6), (18.1, -1.6), (18.1, -1.6)]),
}


@pytest.mark.parametrize("name", sorted(TABLE2))
def test_table2_row(name):
    n, m, amin, cells = TABLE2[name]
    g = redline.load_grid(name)
    assert (g.n, g.m, redline.alpha_min(g)) == (n, m, amin)
    for a, (gain, net) in zip([0.10, 0.15, 0.20], cells):
        i = redline.run(g, a)[1]
        got = (0.0, 0.0) if i == g.ref else (round(g.gain(i), 1), round(g.net(i), 1))
        assert got == (gain, net), (a, g.configs[i])


def test_table2_quantization_row():
    g = redline.load_grid("quant_llama_gsm8k")
    weight_gb = json.loads((DATA / "quant_memory.json").read_text())["weight_gb"]
    assert (g.n, g.m, redline.alpha_min(g)) == (512, 4, 0.07)
    for a in (0.10, 0.15, 0.20):
        i = redline.run(g, a)[1]
        assert g.configs[i] == "nf4" and round(g.net(i), 1) == -1.4
    assert round(weight_gb["bf16"] / weight_gb["nf4"], 2) == 2.81


@pytest.mark.parametrize("name", ["llada2_math", "llada2_code", "sdar_math", "sdar_code", "quant_llama_gsm8k"])
def test_select_agrees_with_the_grid_runner(name):
    g = redline.load_grid(name)
    for a in redline.FINE:
        valid, i, p = redline.run(g, a)
        res = redline.select(g.correct, g.cost(), g.ref, a, lower_is_better=g.lower_is_better)
        assert res.deployed == i and res.valid.tolist() == valid.tolist()


def test_table4_distilled_checkpoint():
    # deployed at every budget from 0.18 on, +46.7% against +39.5% for the best hand-picked config
    g = redline.load_grid("sdar_math500_distilled")
    assert (g.n, g.m) == (500, 16)
    for a in [0.18, 0.19, 0.20]:
        i = redline.run(g, a)[1]
        assert g.configs[i] == "distilled/acc85/semi70" and round(g.gain(i), 1) == 46.7
    i = redline.run(g, 0.15)[1]
    assert g.configs[i] == "acc85/semi70" and round(g.gain(i), 1) == 39.5


def test_table7_calibration_size():
    assert [redline.prompts_needed(0.05, 0.03, 0.10 / m) for m in (8, 16, 50)] == [1008, 1189, 1444]
    assert round(100 * redline.critical_count(1012, 0.10, 0.10 / 8) / 1012, 1) == 7.8


def test_synthetic_check_of_appendix_a7():
    out = subprocess.run([sys.executable, str(ROOT / "experiments" / "synthetic_check.py")],
                         capture_output=True, text=True, check=True, timeout=120).stdout
    assert "family-wise error rate 0.0003" in out
