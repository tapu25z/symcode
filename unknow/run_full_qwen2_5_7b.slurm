#!/bin/bash
#SBATCH --job-name=full_qwen2_5_7b
#SBATCH --output=logs/full_qwen2_5_7b_%j.out
#SBATCH --error=logs/full_qwen2_5_7b_%j.err
#SBATCH --nodelist=DGX-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=mps:a100:2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --export=ALL

# ==============================================================================
# Benchmark Full MATH-500: Qwen/Qwen2.5-7B-Instruct (7B General, 4-bit NF4)
# Model: Qwen/Qwen2.5-7B-Instruct
# Dataset: MATH-500 (Full 500 samples)
# Pipelines: Direct, CoT, SymCode, SymPlanner
# ==============================================================================

export PYTHONUNBUFFERED=1

# Tu dong tim token tu: tham so $1, bien HF_TOKEN, file .hf_token, hoac cache cua huggingface-cli
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

mkdir -p logs
mkdir -p results

echo "⏰ [$(date '+%Y-%m-%d %H:%M:%S')] Start Job $SLURM_JOB_ID on $(hostname)"

# ---------------------------------------------------------
# 1. THIET LAP MOI TRUONG & VENV
# ---------------------------------------------------------
if command -v module &> /dev/null; then
    module clear -f || true
    module load shared python312 || true
fi

if [ -d "/home/uitchain/datastore/uitchain/symcode" ]; then
    export WORK_DIR="/home/uitchain/datastore/uitchain/symcode"
elif [ -d "/datastore/uitchain/symcode" ]; then
    export WORK_DIR="/datastore/uitchain/symcode"
elif [ -d "$HOME/datastore/uitchain/symcode" ]; then
    export WORK_DIR="$HOME/datastore/uitchain/symcode"
else
    export WORK_DIR="$(pwd)"
fi

cd "$WORK_DIR"
echo "📍 [$(date '+%H:%M:%S')] Thu muc lam viec: $WORK_DIR"

export VENV_PATH="$WORK_DIR/venv_a100_8_9"

if [ ! -d "$VENV_PATH" ]; then
    echo "📦 [$(date '+%H:%M:%S')] Dang khoi tao moi truong ao moi tai $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip setuptools wheel --quiet
    echo "📥 Dang cai dat PyTorch CUDA 12.1..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
    echo "📥 Dang cai dat cac thu vien yeu cau tu kaggle/requirements.txt..."
    if [ -f "kaggle/requirements.txt" ]; then
        pip install -r kaggle/requirements.txt --quiet
    else
        pip install "transformers>=4.40.0" "accelerate>=0.28.0" "bitsandbytes>=0.43.0" "sympy>=1.12" "datasets>=2.18.0" tqdm scipy numpy --quiet
    fi
else
    echo "📦 [$(date '+%H:%M:%S')] Kich hoat venv hien huu tai: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi

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
# 2. CHECK GPU CHUAN TU DONG
# ---------------------------------------------------------
echo "🔍 [$(date '+%H:%M:%S')] Dang kiem tra GPU ranh qua gpu_check.sh..."
REQUIRED_VRAM=16384
unset CUDA_VISIBLE_DEVICES

if [ -f "/usr/local/bin/gpu_check.sh" ]; then
    CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 10 ]; then
        echo "⚠️ Het GPU phu hop, Slurm requeue:"
        echo "$CHECK_OUT"
        exit 0
    elif [ $EXIT_CODE -eq 11 ]; then
        echo "❌ Loi kiem tra GPU: $CHECK_OUT"
        exit 1
    fi
    BEST_GPU=$CHECK_OUT
else
    BEST_GPU=0
fi

echo "✅ [$(date '+%H:%M:%S')] GPU duoc chon: $BEST_GPU"

# ---------------------------------------------------------
# 3. KICH HOAT MPS & CUDA VISIBLE DEVICES
# ---------------------------------------------------------
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

export CUDA_VISIBLE_DEVICES=$BEST_GPU

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

echo "🔍 [$(date '+%H:%M:%S')] Kiem tra trang thai GPU PyTorch:"
python3 -c "import torch; print('-> CUDA Available:', torch.cuda.is_available()); print('-> Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ---------------------------------------------------------
# 4. CHAY BENCHMARK FULL 500 MAU MATH-500
# ---------------------------------------------------------
OUTPUT_RESULT="results/results_full_math500_qwen2_5_7b.json"

echo "=========================================================================="
echo "🚀 [$(date '+%H:%M:%S')] Bat dau Benchmark FULL 500 mau MATH-500"
echo "   - Model        : Qwen/Qwen2.5-7B-Instruct (4-bit NF4)"
echo "   - Pipelines    : Direct, CoT, SymCode, SymPlanner"
echo "   - Dataset      : MATH-500 (Full 500 mau)"
echo "   - Output File  : $OUTPUT_RESULT"
echo "=========================================================================="

python kaggle/run_benchmark.py \
    --dataset math500 \
    --model-id "Qwen/Qwen2.5-7B-Instruct" \
    --load-in-4bit \
    --methods Direct CoT SymCode SymPlanner \
    --run-order by-problem \
    --output-file "$OUTPUT_RESULT" \
    --timeout 15 \
    --max-retries 2 \
    --save-every 1

echo "=========================================================================="
echo "🎉 [$(date '+%H:%M:%S')] Hoan tat Benchmark FULL MATH-500 cho Qwen/Qwen2.5-7B-Instruct!"
echo "📁 Ket qua da duoc luu tai: $OUTPUT_RESULT"
echo "=========================================================================="

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
