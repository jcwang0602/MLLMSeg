#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/train_common.sh"

# BATCH_SIZE=128
# PER_DEVICE_BATCH_SIZE=16

MODEL_NAME="OpenGVLab/InternVL3_5-8B"
MODEL_TAG="8b"

run_full_pipeline
