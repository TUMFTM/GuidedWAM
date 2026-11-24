#!/usr/bin/bash
#
# VLM-based scene sampling pipeline for PhysicalAI driving clips.
#
# Runs three stages in order:
#   1) qwvl_scene_select.py     -- tag clips with Qwen3-VL (multi-GPU, resume-safe)
#   2) qwvl_scene_stats.py      -- print stats, save ranked CSV, optional stratified sample
#   3) qwvl_scene_sample.py     -- two-tier train-set subsampling (high-score keep-all + low random)
#
# Override any variable from the command line:
#   NUM_GPUS=8 SPLIT=train bash scripts/drivewam_physicalai_vlm_data_sample.sh
#
# To run only specific stages, set SKIP_STAGE1/2/3=1.

set -euo pipefail

umask 007

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_ROOT=${DATASET_ROOT:-"./data/physicalai/front"}
CLIP_INDEX=${CLIP_INDEX:-"./data/physicalai/clip_index.parquet"}
SCENE_TAGS_DIR=${SCENE_TAGS_DIR:-"./scene_tags"}   # stage 1 output, stages 2+3 input
RANKED_CSV=${RANKED_CSV:-"${SCENE_TAGS_DIR}/ranked.csv"}
TRAIN_SAMPLE_OUT=${TRAIN_SAMPLE_OUT:-"${SCENE_TAGS_DIR}/train_subset.csv"}
TEST_SAMPLE_OUT=${TEST_SAMPLE_OUT:-"${SCENE_TAGS_DIR}/test_sample.csv"}

# ---------------------------------------------------------------------------
# Stage 1 — VLM tagging
# ---------------------------------------------------------------------------
SPLIT=${SPLIT:-"train"}
MODEL_PATH=${MODEL_PATH:-"Qwen3-VL-8B-Instruct"}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_SAMPLE_FRAMES=${NUM_SAMPLE_FRAMES:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-300}
LOADER_WORKERS=${LOADER_WORKERS:-12}
LOAD_STAGGER_S=${LOAD_STAGGER_S:-10}
START_IDX=${START_IDX:-0}    # inclusive; set on each host when sharding across nodes
END_IDX=${END_IDX:--1}       # -1 = process until end of split

# ---------------------------------------------------------------------------
# Stage 2 — Stats
# ---------------------------------------------------------------------------
SAVE_TOP_N=${SAVE_TOP_N:-200}

# ---------------------------------------------------------------------------
# Stage 3a — Train two-tier sampling (used when SPLIT=train)
# ---------------------------------------------------------------------------
SCORE_THRESHOLD=${SCORE_THRESHOLD:-2.0}   # keep ALL clips >= this score
LOW_FRAC=${LOW_FRAC:-0.5}                 # random-sample this fraction of below-threshold clips

# Stage 3b — Stratified test sampling (used when SPLIT=test)
SAMPLE_TOTAL=${SAMPLE_TOTAL:-1000}
SAMPLE_QUANTILE=${SAMPLE_QUANTILE:-0.75}
SAMPLE_RARE_PER_TAG=${SAMPLE_RARE_PER_TAG:-30}
SAMPLE_BASELINE=${SAMPLE_BASELINE:-200}

# ---------------------------------------------------------------------------
# Skip flags
# ---------------------------------------------------------------------------
SKIP_STAGE1=${SKIP_STAGE1:-0}
SKIP_STAGE2=${SKIP_STAGE2:-0}
SKIP_STAGE3=${SKIP_STAGE3:-0}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../src" && pwd)"
cd "${SRC_DIR}"

mkdir -p "${SCENE_TAGS_DIR}"

# ---------------------------------------------------------------------------
# Stage 1: VLM tagging
# ---------------------------------------------------------------------------
if [[ "${SKIP_STAGE1}" == "0" ]]; then
    echo "===== [1/3] qwvl_scene_select.py  (split=${SPLIT}, gpus=${NUM_GPUS}) ====="
    python -m physicalai.vlm_data_sample.qwvl_scene_select \
        --model_path        "${MODEL_PATH}" \
        --dataset_root      "${DATASET_ROOT}" \
        --clip_index        "${CLIP_INDEX}" \
        --split             "${SPLIT}" \
        --output_dir        "${SCENE_TAGS_DIR}" \
        --num_gpus          "${NUM_GPUS}" \
        --batch_size        "${BATCH_SIZE}" \
        --num_sample_frames "${NUM_SAMPLE_FRAMES}" \
        --max_new_tokens    "${MAX_NEW_TOKENS}" \
        --loader_workers    "${LOADER_WORKERS}" \
        --load_stagger_s    "${LOAD_STAGGER_S}" \
        --start_idx         "${START_IDX}" \
        --end_idx           "${END_IDX}"
else
    echo "===== [1/3] SKIPPED (SKIP_STAGE1=1) ====="
fi

# ---------------------------------------------------------------------------
# Stage 2: Stats + ranked CSV
# ---------------------------------------------------------------------------
if [[ "${SKIP_STAGE2}" == "0" ]]; then
    echo "===== [2/3] qwvl_scene_stats.py ====="
    python -m physicalai.vlm_data_sample.qwvl_scene_stats \
        --input_dir    "${SCENE_TAGS_DIR}" \
        --dataset_root "${DATASET_ROOT}" \
        --save_csv     "${RANKED_CSV}" \
        --top_n        "${SAVE_TOP_N}"
else
    echo "===== [2/3] SKIPPED (SKIP_STAGE2=1) ====="
fi

# ---------------------------------------------------------------------------
# Stage 3: Sampling
# ---------------------------------------------------------------------------
if [[ "${SKIP_STAGE3}" == "0" ]]; then
    if [[ "${SPLIT}" == "train" ]]; then
        echo "===== [3/3] qwvl_scene_sample.py  (two-tier train sampling) ====="
        python -m physicalai.vlm_data_sample.qwvl_scene_sample \
            --input_dir      "${SCENE_TAGS_DIR}" \
            --dataset_root   "${DATASET_ROOT}" \
            --out            "${TRAIN_SAMPLE_OUT}" \
            --score_threshold "${SCORE_THRESHOLD}" \
            --low_frac       "${LOW_FRAC}"
    else
        echo "===== [3/3] qwvl_scene_stats.py  (stratified test sampling) ====="
        python -m physicalai.vlm_data_sample.qwvl_scene_stats \
            --input_dir          "${SCENE_TAGS_DIR}" \
            --dataset_root       "${DATASET_ROOT}" \
            --save_csv           "${RANKED_CSV}" \
            --sample_out         "${TEST_SAMPLE_OUT}" \
            --sample_total       "${SAMPLE_TOTAL}" \
            --sample_quantile    "${SAMPLE_QUANTILE}" \
            --sample_rare_per_tag "${SAMPLE_RARE_PER_TAG}" \
            --sample_baseline    "${SAMPLE_BASELINE}" \
            --no_print_events
    fi
else
    echo "===== [3/3] SKIPPED (SKIP_STAGE3=1) ====="
fi

echo "===== Done. Scene tags -> ${SCENE_TAGS_DIR} ====="
