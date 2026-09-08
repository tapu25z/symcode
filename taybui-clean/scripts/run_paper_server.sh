#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python_bin="${PAPER_PYTHON:-/opt/symplan-paper-v3-venv/bin/python}"
exec "$python_bin" -u -m paperbench.pipeline --model qwen3-4b \
  --executor "${PAPER_EXECUTOR:-process}" \
  --output "${PAPER_RUN_ROOT:-paper_runs/paper_v3}" "$@"
