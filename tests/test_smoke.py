import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import redline

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "tests" / "fixtures" / "toy_outcomes.csv"


def run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          check=True, timeout=120).stdout


def test_import_stays_light():
    # a fresh interpreter, since pytest plugins may have imported more already
    out = run("-c", "import sys, redline; print('matplotlib' in sys.modules, 'torch' in sys.modules)")
    assert out.strip() == "False False"


def test_pvalue_is_the_binomial_tail():
    assert redline.pvalues(1, 20, 0.3) == pytest.approx(0.7 ** 20 + 20 * 0.3 * 0.7 ** 19)


def test_holm_steps_down_and_stops_at_the_first_failure():
    assert redline.holm([0.01, 0.04, 0.03], 0.10).tolist() == [True, True, True]
    # 0.02 <= 0.1/3 passes, then 0.06 > 0.1/2 stops the walk, so 0.07 is never tested
    assert redline.holm([0.02, 0.06, 0.07], 0.10).tolist() == [True, False, False]


def test_holm_keeps_everything_bonferroni_keeps():
    p = np.random.default_rng(0).random((200, 6)) ** 3
    for row in p:
        assert np.all(redline.holm(row, 0.10) >= (row <= 0.10 / len(row)))


def test_select_on_a_hand_checked_grid():
    g = redline.load_outcomes(str(TOY), reference="default")
    res = redline.select(g.correct, g.cost(), reference=g.ref, alpha=0.30, names=g.configs)
    # k = 0, 1, 6 of 20; only "fastest" fails its test
    assert res.violations.tolist() == [0, 1, 6]
    assert res.valid.tolist() == [True, True, False]
    assert res.deployed_name == "fast"
    assert res.gain == pytest.approx(25.0)


def test_select_deploys_the_reference_when_nothing_passes():
    g = redline.load_outcomes(str(TOY), reference="default")
    res = redline.select(g.correct, g.cost(), reference=g.ref, alpha=0.05, names=g.configs)
    assert res.deployed_name == "default"
    assert "nothing faster is valid" in str(res)


def test_quickstart_prints_what_the_readme_shows():
    out = run(str(ROOT / "examples" / "quickstart.py"))
    assert "D               6.40  0.750  0.134  1.0e+00  no" in out
    assert "deploy C, +45.0% over the reference" in out


def test_demo_is_the_quickstart():
    assert run("-m", "redline", "demo") == run(str(ROOT / "examples" / "quickstart.py"))


def test_cli_help():
    out = run("-m", "redline", "--help")
    for cmd in ("demo", "select", "sample-size"):
        assert cmd in out


def test_cli_select_writes_json(tmp_path):
    out = tmp_path / "res.json"
    run("-m", "redline", "select", str(TOY), "--reference", "default", "--alpha", "0.30", "--json", str(out))
    res = json.loads(out.read_text())
    assert res["deployed"] == "fast"
    assert [c["valid"] for c in res["configs"]] == [True, True, False]


def test_cli_sample_size_matches_table7():
    out = run("-m", "redline", "sample-size", "--alpha", "0.05", "--risk", "0.03", "--m", "8")
    assert out.startswith("1008 prompts")
