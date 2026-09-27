"""Table 2: the configuration deployed at alpha = 0.10, 0.15, 0.20 for each grid."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from redline import DATA_DIR, DELTA, alpha_min, load_grid, run  # noqa: E402

ROWS = ["llada2_math", "llada2_math_large", "llada2_code", "sdar_math", "sdar_code",
        "specdec_llama_gsm8k", "specdec_qwen_gsm8k", "quant_llama_gsm8k"]


def gain_str(g, i, weight_gb):
    if g.kind == "memory":
        # measured weight memory where available, sizes from the bit width otherwise
        name, ref = g.configs[i], g.configs[g.ref]
        if name in weight_gb and ref in weight_gb:
            return f"{weight_gb[ref] / weight_gb[name]:.2f}x"
        return f"{g.gain(i):.2f}x"
    return "0.0" if i == g.ref else f"{g.gain(i):+.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--grids", default=",".join(ROWS))
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--alphas", default="0.10,0.15,0.20")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    with open(os.path.join(args.data_dir, "quant_memory.json")) as f:
        weight_gb = json.load(f)["weight_gb"]

    print(f"delta = {args.delta}, gain in % (TPF or accepted tokens per target forward) or memory reduction, "
          "net accuracy change in points")
    head = f"{'grid':32s} {'n':>5s} {'m':>3s} {'a_min':>5s}"
    for a in alphas:
        head += f" | {'a=%.2f' % a:22s} {'gain':>6s} {'net':>5s}"
    print(head)
    for name in args.grids.split(","):
        g = load_grid(name, args.data_dir)
        am = alpha_min(g, args.delta)
        line = f"{g.label:32s} {g.n:5d} {g.m:3d} {'-' if am is None else '%.2f' % am:>5s}"
        for a in alphas:
            _, i, _ = run(g, a, args.delta)
            net = "0.0" if i == g.ref else f"{g.net(i):+.1f}"
            line += f" | {g.configs[i]:22s} {gain_str(g, i, weight_gb):>6s} {net:>5s}"
        print(line)


if __name__ == "__main__":
    main()
