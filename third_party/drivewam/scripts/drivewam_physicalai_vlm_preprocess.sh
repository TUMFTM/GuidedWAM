#!/usr/bin/bash
#
# Run the per-chunk navigation-guidance preprocessing pipeline in order:
#   1) generate_route.py                  -> route_store.json
#   2) generate_bev.py                    -> bev_store.npz
#   3) generate_scene_evolving_guidance.py -> guidance JSON (consumes 1 + 2)
#
# Override any of the env vars below from the command line, e.g.:
#   DATASET_ROOT=/path/to/data OUTPUT_DIR=/path/to/out bash scripts/drivewam_physicalai_vlm_preprocess.sh

set -euo pipefail

umask 007

DATASET_ROOT=${DATASET_ROOT:-"./data/physicalai/front"}
CLIP_INDEX=${CLIP_INDEX:-"./data/physicalai/clip_index.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"./preprocess_out"}
CLIP_IDS_FILE=${CLIP_IDS_FILE:-""}
NUM_CLIPS=${NUM_CLIPS:-"-1"}
NUM_CHUNKS=${NUM_CHUNKS:-"5"}
CHUNK_SECONDS=${CHUNK_SECONDS:-"4.0"}
NUM_WORKERS=${NUM_WORKERS:-"16"}
SEED=${SEED:-"0"}

ROUTE_STORE_NAME=${ROUTE_STORE_NAME:-"route_store.json"}
BEV_STORE_NAME=${BEV_STORE_NAME:-"bev_store.npz"}
GUIDANCE_OUTPUT_NAME=${GUIDANCE_OUTPUT_NAME:-"scene_evolving_guidance.json"}

VLM_MODEL_PATH=${VLM_MODEL_PATH:-"Qwen3-VL-8B-Instruct"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-"8"}
BATCH_SIZE=${BATCH_SIZE:-"64"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"32768"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.9"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../src" && pwd)"
cd "${SRC_DIR}"

mkdir -p "${OUTPUT_DIR}"

common_args=(
    --dataset_root "${DATASET_ROOT}"
    --clip_index "${CLIP_INDEX}"
    --output_dir "${OUTPUT_DIR}"
    --num_clips "${NUM_CLIPS}"
    --num_chunks "${NUM_CHUNKS}"
    --chunk_seconds "${CHUNK_SECONDS}"
    --seed "${SEED}"
)
if [[ -n "${CLIP_IDS_FILE}" ]]; then
    common_args+=(--clip_ids_file "${CLIP_IDS_FILE}")
fi

echo "===== [1/3] generate_route.py ====="
python -m physicalai.vlm_preprocess.generate_route \
    "${common_args[@]}" \
    --num_workers "${NUM_WORKERS}" \
    --route_store_name "${ROUTE_STORE_NAME}"

echo "===== [2/3] generate_bev.py ====="
python -m physicalai.vlm_preprocess.generate_bev \
    "${common_args[@]}" \
    --num_workers "${NUM_WORKERS}" \
    --bev_store_name "${BEV_STORE_NAME}"

echo "===== [3/3] generate_scene_evolving_guidance.py ====="
python -m physicalai.vlm_preprocess.generate_scene_evolving_guidance \
    "${common_args[@]}" \
    --model_path "${VLM_MODEL_PATH}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --route_store "${OUTPUT_DIR}/${ROUTE_STORE_NAME}" \
    --bev_store "${OUTPUT_DIR}/${BEV_STORE_NAME}" \
    --output_name "${GUIDANCE_OUTPUT_NAME}"

echo "===== Done. Guidance JSON -> ${OUTPUT_DIR}/${GUIDANCE_OUTPUT_NAME} ====="
