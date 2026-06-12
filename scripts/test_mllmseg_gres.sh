#!/bin/bash
# Evaluate GRES checkpoints on gRefCOCO.
#
# Usage:
#   CHECKPOINT=work_dirs/... bash scripts/test_mllmseg_gres.sh

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
    -m mllmseg.test_mllmseg_gres \
    --checkpoint "${CHECKPOINT}" \
    --base_image_dir "${PROJECT_ROOT}/data"
