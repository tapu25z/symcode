#!/usr/bin/env bash
# ==============================================================================
# Vast.ai Benchmark Execution Script for RTX 3090 (CUDA 12.4)
# ==============================================================================
# Usage:
#   chmod +x run_vastai.sh
#   ./run_vastai.sh                         # Run default benchmark
#   ./run_vastai.sh --num-samples 50         # Run 50 samples
#   ./run_vastai.sh --model-id "Qwen/Qwen2.5-Math-7B-Instruct"
# ==============================================================================

set -e

# Export environment variables
export CUDA_VISIBLE_DEVICES=0
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
    echo "[WARN] nvidia-smi not found. Ensure NVIDIA driver is loaded properly."
fi

# 2. Check & install Python dependencies
echo "[INFO] Checking Python dependencies..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || {
    echo "[INFO] Installing PyTorch for CUDA 12.4..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
}

if [ -f "kaggle/requirements.txt" ]; then
    echo "[INFO] Installing requirements from kaggle/requirements.txt..."
    pip install -q -r kaggle/requirements.txt
fi

# 3. Create results directory
mkdir -p results

# 4. Default execution parameters (can be overridden via CLI args)
DEFAULT_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_DATASET="math500"
DEFAULT_METHODS="Direct CoT SymCode SymPlanner"

echo "======================================================================"
echo " Executing Benchmark Evaluation"
echo "======================================================================"

if [ "$#" -eq 0 ]; then
    echo "[INFO] No custom arguments provided. Running default benchmark configuration:"
    echo "       Model:   ${DEFAULT_MODEL}"
    echo "       Dataset: ${DEFAULT_DATASET}"
    echo "       Methods: ${DEFAULT_METHODS}"
    echo "       4-Bit:   Enabled (NF4 via bitsandbytes)"
    echo ""
    python3 kaggle/run_benchmark.py \
        --dataset "${DEFAULT_DATASET}" \
        --methods ${DEFAULT_METHODS} \
        --model-id "${DEFAULT_MODEL}" \
        --load-in-4bit
else
    echo "[INFO] Passing custom arguments to kaggle/run_benchmark.py: $@"
    python3 kaggle/run_benchmark.py "$@"
fi

echo "======================================================================"
echo " Benchmark Execution Finished Successfully!"
echo "======================================================================"
