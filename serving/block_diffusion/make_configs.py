"""Write engine experiment configs for the block-diffusion grids (one yaml per grid)."""
import argparse
import itertools
import os

FAMILY = {
    "llada2": dict(model_name="llada2_mini", mask_token_id=156895, buffer_size=2),
    "sdar": dict(model_name="sdar", mask_token_id=151669, buffer_size=4),
}
TASKS = {
    ("llada2", "math"): "gsm8k_llada2,math500_llada2",
    ("llada2", "code"): "humaneval_plus_llada2,mbpp_plus_llada2",
    ("sdar", "math"): "gsm8k_sdar,math500_sdar",
    ("sdar", "code"): "humaneval_plus_sdar,mbpp_sdar",
}
MAIN = [(a, s, 0.10) for a, s in itertools.product([0.85, 0.90, 0.95, 0.99], [0.70, 0.90])]
LARGE = list(itertools.product([0.75, 0.80, 0.85, 0.90, 0.95], [0.5, 0.6, 0.7, 0.8, 0.9], [0.10, 0.30]))
GRIDS = {
    # the llada2 math grid has no acc99/semi70
    "llada2_math": ("llada2", "math", [c for c in MAIN if c[:2] != (0.99, 0.70)]),
    "llada2_math_add": ("llada2", "math", [(0.95, 0.90, 0.05), (0.95, 0.90, 0.20)]),
    # the reduced math prompt sets (256 and 32 per subtask) keep all eight
    "llada2_math_small": ("llada2", "math", MAIN),
    "llada2_math_large": ("llada2", "math", LARGE),
    "llada2_code": ("llada2", "code", MAIN),
    "sdar_math": ("sdar", "math", MAIN),
    "sdar_code": ("sdar", "code", MAIN),
}

EXP = """  - name: "{name}"
    task: "grid_task"
    model: "grid_model"
    decoding_strategy: "multi_bd"
    sampling_mode: "naive"
    block_size: 32
    page_size: 32
    buffer_size: {buffer_size}
    engine:
      master_port: {port}
      tensor_parallel_size: 1
    thresholds:
      add_block_threshold: {add}
      semi_complete_threshold: {semi}
      accept_threshold: {acc}
"""


def tag(acc, semi, add, grid):
    # add appears in the name when it is not the engine default or the grid varies it
    t = f"acc{round(acc * 100)}_semi{round(semi * 100)}"
    if add != 0.10 or grid == "llada2_math_large":
        t += f"_add{round(add * 100):02d}"
    return t


def write(name, out_dir, port0):
    fam, task, cells = GRIDS[name]
    f = FAMILY[fam]
    text = (f'name: "{name}"\n\nmodels:\n  grid_model:\n    path: "$MODEL_PATH"\n    env: "MODEL_PATH"\n'
            f'    model_name: "{f["model_name"]}"\n    mask_token_id: {f["mask_token_id"]}\n\n'
            f'tasks:\n  grid_task: "{TASKS[(fam, task)]}"\n\nexperiments:\n')
    for i, (acc, semi, add) in enumerate(cells):
        text += EXP.format(name=f"{name}_{tag(acc, semi, add, name)}", buffer_size=f["buffer_size"], port=port0 + i,
                           add=add, semi=semi, acc=acc)
    path = os.path.join(out_dir, f"{name}.yml")
    with open(path, "w") as fh:
        fh.write(text)
    print(f"wrote {path} ({len(cells)} configurations)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="configs")
    ap.add_argument("--grids", default=",".join(GRIDS))
    ap.add_argument("--port", type=int, default=29000, help="first master port, one per configuration")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    port = args.port
    for name in args.grids.split(","):
        write(name, args.out_dir, port)
        port += len(GRIDS[name][2])


if __name__ == "__main__":
    main()
