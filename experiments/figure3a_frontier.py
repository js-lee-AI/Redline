"""Figure 3a: deployed gain over the budget grid 0.01..0.30 for the four block-diffusion grids.

With several --deltas, also counts the (grid, budget, delta) evaluations whose deployment differs from
the first delta (sensitivity to delta, Appendix D.3).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from redline import DATA_DIR, FINE, load_grid, run  # noqa: E402

GRIDS = ["llada2_math", "sdar_math", "llada2_code", "sdar_code"]


def curve(g, delta):
    out = []
    for a in FINE:
        i = run(g, a, delta)[1]
        out.append((a, g.configs[i], g.gain(i)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--grids", default=",".join(GRIDS))
    ap.add_argument("--deltas", default="0.10")
    ap.add_argument("--plot", default=None, help="optional output figure, e.g. frontier.pdf")
    args = ap.parse_args()
    deltas = [float(d) for d in args.deltas.split(",")]
    grids = [load_grid(n, args.data_dir) for n in args.grids.split(",")]
    curves = {d: [curve(g, d) for g in grids] for d in deltas}
    main_curves = curves[deltas[0]]

    print(f"deployed TPF gain (%) by budget, delta = {deltas[0]}")
    print(f"{'alpha':>5s}" + "".join(f" {g.label:>24s}" for g in grids))
    for j, a in enumerate(FINE):
        print(f"{a:5.2f}" + "".join(f" {c[j][1]:>16s} {c[j][2]:+6.1f}" for c in main_curves))
    for g, c in zip(grids, main_curves):
        first = next((x for x in c if x[1] != g.configs[g.ref]), None)
        if first:
            print(f"{g.label}: first gain at alpha = {first[0]:.2f} ({first[1]}, {first[2]:+.1f}%)")
        else:
            print(f"{g.label}: reference at every budget")

    if len(deltas) > 1:
        changed = total = 0
        for d in deltas:
            for c0, c in zip(main_curves, curves[d]):
                total += len(c)
                changed += sum(x[1] != y[1] or abs(x[2] - y[2]) > 1e-6 for x, y in zip(c0, c))
        j = FINE.index(0.10)
        for d in deltas:
            print(f"delta = {d}: at alpha = 0.10 " + ", ".join(
                f"{g.label} {c[j][1]} {c[j][2]:+.1f}%" for g, c in zip(grids, curves[d])))
        print(f"{changed} of {total} evaluations (grids x budgets x deltas) deploy differently than at "
              f"delta = {deltas[0]}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for g, c in zip(grids, main_curves):
            ax.step([0.0] + FINE, [0.0] + [x[2] for x in c], where="post", label=g.label)
        ax.set_xlim(0, 0.30)
        ax.set_xlabel("risk budget alpha")
        ax.set_ylabel("deployed TPF gain (%)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
