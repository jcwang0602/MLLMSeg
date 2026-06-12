#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/train_common.sh"

MODEL_NAME="OpenGVLab/InternVL3_5-4B"
MODEL_TAG="4b"

run_full_pipeline
