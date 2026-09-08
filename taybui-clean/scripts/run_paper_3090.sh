#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
phase="${1:-smoke}"
model="${MODEL_PRESET:-qwen3-4b}"
python_bin="${PAPER_PYTHON:-python3}"
executor="${PAPER_EXECUTOR:-process}"
run_root="${PAPER_RUN_ROOT:-paper_runs/manual_v3}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dataset() {
  "$python_bin" -m paperbench.run --model "$model" --dataset "$1" \
    --phase "$phase" --executor "$executor" \
    --output "$run_root/${model}_${1}_${phase}" "${@:2}"
}

case "$phase" in
  smoke)
    run_dataset math500 --limit 8
    ;;
  dev)
    run_dataset math500
    run_dataset gsm8k
    ;;
  freeze)
    "$python_bin" -m paperbench.freeze \
      --dev-runs "$run_root/${model}_math500_dev" "$run_root/${model}_gsm8k_dev" \
      --output "$run_root/protocol.lock.json"
    ;;
  test)
    run_dataset math500 --lock "$run_root/protocol.lock.json"
    run_dataset gsm8k --lock "$run_root/protocol.lock.json"
    ;;
  *) echo "Usage: $0 smoke|dev|freeze|test" >&2; exit 2 ;;
esac
