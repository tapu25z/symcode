#!/usr/bin/env bash
# ==============================================================================
# Vast.ai Benchmark Execution Script for RTX 3090 (CUDA 12.4)
# Model: Qwen/Qwen2.5-Coder-7B-Instruct (4-bit NF4 Quantization)
# ==============================================================================
# Usage:
#   chmod +x unknow/run_vastai.sh
#   ./unknow/run_vastai.sh                     # Run default benchmark (Qwen/Qwen2.5-Coder-7B-Instruct 4-bit)
#   ./unknow/run_vastai.sh --num-samples 50     # Run 50 samples
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# Export environment variables
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "======================================================================"
echo " Starting Benchmark Setup on Vast.ai (RTX 3090 / CUDA 12.4)"
echo "======================================================================"

# 1. Check GPU status
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] GPU Status:"
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
else
    echo "[ERROR] NVIDIA GPU required for this 4-bit benchmark." >&2
    exit 1
fi

# 2. Check & install Python dependencies
echo "[INFO] Checking Python dependencies..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || {
    echo "[INFO] Installing PyTorch for CUDA 12.4..."
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
}

if [ -f "kaggle/requirements.txt" ]; then
    echo "[INFO] Installing requirements from kaggle/requirements.txt..."
    python3 -m pip install -q -r kaggle/requirements.txt "transformers>=4.56,<5"
fi

# Stop before downloading the model if PyTorch cannot use CUDA.
python3 - <<'CUDA_CHECK'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] PyTorch cannot use CUDA. Use a Vast.ai PyTorch CUDA image and check the NVIDIA driver.")
print(f"[INFO] CUDA GPU: {torch.cuda.get_device_name(0)}")
CUDA_CHECK

# 3. Create results & logs directories
mkdir -p results logs

# 4. Default execution parameters
DEFAULT_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_DATASET="math500"
DEFAULT_METHODS="Direct CoT SymCode SymPlanner"
SAMPLE_FILE="kaggle/data/math500/test_50_stratified.jsonl"
OUTPUT_RESULT="results/results_vastai_qwen2_5_coder_7b_4bit.json"

echo "======================================================================"
echo " Executing Benchmark Evaluation"
echo "======================================================================"

echo "[INFO] Model: ${DEFAULT_MODEL} | Quantization: 4-bit NF4"
echo "[INFO] Default dataset: ${SAMPLE_FILE} (50 stratified samples)"
echo "[INFO] Additional arguments override the defaults: $*"
python3 kaggle/run_benchmark.py \
    --dataset "${DEFAULT_DATASET}" \
    --dataset-path "${SAMPLE_FILE}" \
    --methods ${DEFAULT_METHODS} \
    --model-id "${DEFAULT_MODEL}" \
    --load-in-4bit \
    --run-order by-problem \
    --output-file "${OUTPUT_RESULT}" \
    --timeout 15 \
    --max-retries 2 \
    --save-every 1 \
    "$@"

echo "======================================================================"
echo " Benchmark Execution Finished Successfully!"
echo "======================================================================"
