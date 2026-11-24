#!/usr/bin/bash

set -x

umask 007

MASTER_PORT=${MASTER_PORT:-"29501"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"drivewam_navsim_train"}
export WANDB_API_KEY=""
export WANDB_TEAM_NAME=""
export WANDB_PROJECT="drivewam"

## node setting
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \

torchrun \
    --nproc_per_node=8 \
    --master_port ${master_port} \
    --tee 3 \
    -m src.navsim.train