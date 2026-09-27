"""Per-prompt outcome file (config, prompt, correct, tokens, forwards) from served grid runs.

Correctness is the engine's own task score (exact_match in the lm-eval samples, execution-based for code).
Tokens and forwards of each request come from the decode trajectory, and the sums are checked against
the engine's stats file.
"""
import argparse
import csv
import glob
import json
import os
import re

KNOBS = re.compile(r"acc(\d+)_semi(\d+)(?:_add(\d+))?")


def label(dirname, prefix=""):
    m = KNOBS.search(dirname)
    s = f"acc{m.group(1)}/semi{m.group(2)}"
    if m.group(3):
        s += f"/add{int(m.group(3)) / 100:.2f}"
    return prefix + s


def find(exp_dir, pattern):
    hits = glob.glob(os.path.join(exp_dir, "**", pattern), recursive=True)
    assert len(hits) == 1, (exp_dir, pattern, len(hits))
    return hits[0]


def load_run(exp_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "**", "samples_*.jsonl"), recursive=True)):
        task = os.path.basename(path).split("samples_")[1].rsplit("_2", 1)[0]  # drop the timestamp
        task = re.sub(r"_(llada2|sdar)$", "", task)
        with open(path, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                em = o.get("exact_match")
                if em is None:
                    em = o["metrics"]["exact_match"]
                rows.append((task, int(o["doc_id"]), o["resps"][0][0], int(round(float(em)))))
    with open(find(exp_dir, "0x1_truncated_responses.json")) as f:
        responses = json.load(f)
    with open(find(exp_dir, "0x3_decode_trajectory.json")) as f:
        cost = [(t["req_id"], len(t["token_ids"]), len(t["trajectory"])) for t in json.load(f)]
    with open(os.path.join(exp_dir, "diffulex_stats.json")) as f:
        stats = json.load(f)
    # requests are served in samples order, check it before pairing scores with costs
    assert len(rows) == len(responses) == len(cost), exp_dir
    assert all(r[2] == responses[i] for i, r in enumerate(rows)), f"request order differs in {exp_dir}"
    assert all(c[0] == i for i, c in enumerate(cost)), exp_dir
    assert sum(c[1] for c in cost) == stats["total_tokens"] and sum(c[2] for c in cost) == stats["total_nfe"]
    return {(r[0], r[1]): (r[3], cost[i][1], cost[i][2]) for i, r in enumerate(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="output dirs of run_grid.sh")
    ap.add_argument("--out", required=True, help="csv to write")
    ap.add_argument("--prefix", default="", help="label prefix, e.g. distilled/")
    ap.add_argument("--append", action="store_true", help="add configs to an existing file")
    args = ap.parse_args()
    table = {}
    for run in args.runs:
        for d in sorted(glob.glob(os.path.join(run, "runs", "*"))):
            if glob.glob(os.path.join(d, "**", "samples_*.jsonl"), recursive=True):
                table[label(os.path.basename(d), args.prefix)] = load_run(d)
                print("read", d)
    keys = sorted(set.intersection(*[set(t) for t in table.values()]))
    prompts = [f"{t}/{i}" for t, i in keys]
    if args.append:
        with open(args.out) as f:
            old = [r["prompt"] for r in csv.DictReader(f) if r]
        assert old[:len(prompts)] == prompts, "prompt set differs from the existing file"
    with open(args.out, "a" if args.append else "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if not args.append:
            w.writerow(["config", "prompt", "correct", "tokens", "forwards"])
        for c, t in table.items():
            for k, p in zip(keys, prompts):
                w.writerow([c, p, *t[k]])
    print(f"wrote {args.out}: {len(table)} configurations x {len(prompts)} prompts")


if __name__ == "__main__":
    main()
