#!/bin/bash
# Evaluate RES checkpoints on RefCOCO / RefCOCO+ / RefCOCOg.
#
# Usage:
#   CHECKPOINT=work_dirs/mllmseg_8b_lora64_... bash scripts/test_mllmseg.sh
#   CHECKPOINT=jcwang0602/MLLMSeg_InternVL3_5_8B_RES bash scripts/test_mllmseg.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/train_common.sh"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the model directory to evaluate}"
GPUS="${GPUS:-8}"

setup_env

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m mllmseg.test_mllmseg_res \
    --checkpoint "${CHECKPOINT}" \
    --base_image_dir "${PROJECT_ROOT}/data"
