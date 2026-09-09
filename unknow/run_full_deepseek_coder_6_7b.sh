#!/usr/bin/env bash
# ==============================================================================
# Vast.ai Benchmark Execution Script for RTX 3090 (CUDA 12.4)
# Full MATH-500: deepseek-ai/deepseek-coder-6.7b-instruct (4-bit NF4)
# Model: deepseek-ai/deepseek-coder-6.7b-instruct
# ==============================================================================
# Usage:
#   chmod +x unknow/run_full_deepseek_coder_6_7b.sh
#   ./unknow/run_full_deepseek_coder_6_7b.sh                 # Chay mac dinh
#   ./unknow/run_full_deepseek_coder_6_7b.sh <HF_TOKEN>      # Truyen truc tiep HF_TOKEN
# ==============================================================================

set -eo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---------------------------------------------------------
# 1. TU DONG TIM VA NAP HUGGINGFACE TOKEN
# ---------------------------------------------------------
TOKEN_CANDIDATE="${1:-$HF_TOKEN}"

if [ -z "$TOKEN_CANDIDATE" ]; then
    if [ -f "$HOME/.cache/huggingface/token" ]; then
        TOKEN_CANDIDATE=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null | tr -d '\r\n')
    elif [ -f ".hf_token" ]; then
        TOKEN_CANDIDATE=$(cat ".hf_token" 2>/dev/null | tr -d '\r\n')
    elif [ -f "$HOME/.hf_token" ]; then
        TOKEN_CANDIDATE=$(cat "$HOME/.hf_token" 2>/dev/null | tr -d '\r\n')
    fi
fi

if [ -n "$TOKEN_CANDIDATE" ]; then
    export HF_TOKEN="$TOKEN_CANDIDATE"
    export HUGGING_FACE_HUB_TOKEN="$TOKEN_CANDIDATE"
fi

# ---------------------------------------------------------
# 2. KIEM TRA GPU NVIDIA RTX 3090 & CUDA 12.4
# ---------------------------------------------------------
echo "======================================================================"
echo "🚀 Khoi chay Benchmark tren Vast.ai (RTX 3090 / CUDA 12.4)"
echo "   Model: deepseek-ai/deepseek-coder-6.7b-instruct"
echo "======================================================================"

if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] Thong tin GPU:"
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
else
    echo "[WARN] Khong tim thay nvidia-smi. Hay dam bao NVIDIA driver da cai dat dung."
fi

# ---------------------------------------------------------
# 3. KIEM TRA & CAI DAT PYTORCH CUDA 12.4 VA DEPENDENCIES
# ---------------------------------------------------------
echo "[INFO] Kiem tra PyTorch va CUDA..."
python3 -c "import torch; assert torch.cuda.is_available(); print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Device: {torch.cuda.get_device_name(0)}')" 2>/dev/null || {
    echo "[INFO] Dang cai dat / cap nhat PyTorch CUDA 12.4 cho RTX 3090..."
    pip install --upgrade pip --quiet
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
}

if [ -f "kaggle/requirements.txt" ]; then
    echo "[INFO] Cai dat thu vien tu kaggle/requirements.txt..."
    pip install -q -r kaggle/requirements.txt
else
    pip install "transformers>=4.40.0" "accelerate>=0.28.0" "bitsandbytes>=0.43.0" "sympy>=1.12" "datasets>=2.18.0" tqdm scipy numpy --quiet
fi

# ---------------------------------------------------------
# 4. LOGIN HUGGINGFACE (CAN CHO LLAMA HOAC MODEL CO GATED ACCESS)
# ---------------------------------------------------------
if [ -n "$HF_TOKEN" ]; then
    python3 -c "
try:
    import os
    from huggingface_hub import login
    login(os.environ['HF_TOKEN'])
    print('✅ Dang nhap HuggingFace thanh cong.')
except Exception as ex:
    print(f'⚠️ Loi dang nhap HuggingFace: {ex}')
"
fi

# ---------------------------------------------------------
# 5. THIET LAP THU MUC KET QUA VA LOG
# ---------------------------------------------------------
mkdir -p results logs
OUTPUT_RESULT="results/results_full_math500_deepseek_coder_6_7b.json"
LOG_FILE="logs/run_full_deepseek_coder_6_7b_$(date '+%Y%m%d_%H%M%S').log"

echo "======================================================================"
echo "🎯 Bat dau chay Benchmark: Full MATH-500: deepseek-ai/deepseek-coder-6.7b-instruct (4-bit NF4)"
echo "   - Model        : deepseek-ai/deepseek-coder-6.7b-instruct"
echo "   - Output File  : $OUTPUT_RESULT"
echo "   - Log File     : $LOG_FILE"
echo "======================================================================"

python3 kaggle/run_benchmark.py \
    --dataset math500 --run-order by-problem \
    --model-id "deepseek-ai/deepseek-coder-6.7b-instruct" \
    --load-in-4bit \
    --methods Direct CoT SymCode SymPlanner \
    --output-file "$OUTPUT_RESULT" \
    --timeout 15 \
    --max-retries 2 \
    --save-every 1 2>&1 | tee -a "$LOG_FILE"

echo "======================================================================"
echo "🎉 Hoan tat Benchmark cho deepseek-ai/deepseek-coder-6.7b-instruct!"
echo "📁 File ket qua: $OUTPUT_RESULT"
echo "📁 File log:     $LOG_FILE"
echo "======================================================================"
