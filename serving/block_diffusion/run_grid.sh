#!/usr/bin/env bash
# serve one grid with the engine's batch runner (one GPU per run, nothing else on the card)
# usage: ENGINE_DIR=/path/to/Diffulex MODEL_PATH=/path/to/checkpoint \
#        bash run_grid.sh configs/sdar_math.yml runs/sdar_math [prompts_per_subtask] [max_num_reqs]
set -euo pipefail
: "${ENGINE_DIR:?set ENGINE_DIR to the engine checkout}"
: "${MODEL_PATH:?set MODEL_PATH to the model checkpoint}"
export MODEL_PATH
config=$(realpath "$1")
out=$(realpath -m "$2")
limit=${3:-512}   # 256 and 32 for the reduced llada2 grids
reqs=${4:-8}      # 16 for llada2_code and llada2_math_small
mkdir -p "$out"
cd "$ENGINE_DIR"
OUTPUT_BASE=$(dirname "$out") RUN_ID=$(basename "$out") LOG_DIR="$out/logs" \
CONFIG_DIR=$(dirname "$config") CONFIG_FILES=$(basename "$config") \
DEFAULTS_CONFIG="$ENGINE_DIR/diffulex_bench/configs/experiment/_defaults.yml" \
DATASET_LIMIT="$limit" MAX_NUM_REQS="$reqs" SKIP_MISSING_MODELS=1 HF_ALLOW_CODE_EVAL=1 \
    bash script/run_batch_experiments.sh
