#!/bin/bash
# Download InternVL3.5 base models from Hugging Face into ./models/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
mkdir -p models

python -m huggingface_hub.cli.hf download OpenGVLab/InternVL3_5-1B --local-dir models/InternVL3_5-1B
python -m huggingface_hub.cli.hf download OpenGVLab/InternVL3_5-2B --local-dir models/InternVL3_5-2B
python -m huggingface_hub.cli.hf download OpenGVLab/InternVL3_5-4B --local-dir models/InternVL3_5-4B
python -m huggingface_hub.cli.hf download OpenGVLab/InternVL3_5-8B --local-dir models/InternVL3_5-8B
