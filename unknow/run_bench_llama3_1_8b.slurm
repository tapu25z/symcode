#!/bin/bash
#SBATCH --job-name=sym_llama3_1_8b
#SBATCH --output=logs/bench_llama3_1_8b_%j.out
#SBATCH --error=logs/bench_llama3_1_8b_%j.err
#SBATCH --nodelist=DGX-A100            # Chỉ định chạy trên Node DGX-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16             # 16 CPU cores
#SBATCH --gres=mps:a100:2              # Sử dụng 2 slot MPS trên GPU A100
#SBATCH --mem=32G                      # 32GB RAM hệ thống
#SBATCH --time=12:00:00                # Giới hạn thời gian chạy (12 giờ)
#SBATCH --export=ALL                  # Đảm bảo Slurm nhận toàn bộ biến môi trường từ terminal

# ==============================================================================
# Script Slurm Benchmark: Llama-3.1-8B-Instruct trên 50 mẫu MATH-500
# Đánh giá 4 Phương pháp: Direct, CoT, SymCode, SymPlanner
# Tập dữ liệu: MATH-500 (50 mẫu ngẫu nhiên: 35 câu L1-3, 15 câu L4-5)
# Model: meta-llama/Llama-3.1-8B-Instruct (4-bit NF4 Quantization)
# ==============================================================================
# Cách chạy (chọn 1 trong 3 cách):
#   Cách 1 (Khuyên dùng): export HF_TOKEN="hf_..." && sbatch unknow/run_bench_llama3_1_8b.slurm
#   Cách 2: sbatch unknow/run_bench_llama3_1_8b.slurm "hf_..."
#   Cách 3: huggingface-cli login --token "hf_..." (chỉ cần chạy 1 lần trên terminal)
# ==============================================================================

export PYTHONUNBUFFERED=1

# Tự động tìm token từ: tham số $1, biến HF_TOKEN, file .hf_token, hoặc cache của huggingface-cli
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

if [ -z "$TOKEN_CANDIDATE" ]; then
    echo "⚠️ LỖI: Không tìm thấy HuggingFace Token để tải mô hình meta-llama/Llama-3.1-8B-Instruct."
    echo "   Vui lòng thực hiện một trong các cách sau:"
    echo "   1) export HF_TOKEN=\"hf_...\" && sbatch unknow/run_bench_llama3_1_8b.slurm"
    echo "   2) sbatch unknow/run_bench_llama3_1_8b.slurm \"hf_...\""
    echo "   3) echo \"hf_...\" > .hf_token && sbatch unknow/run_bench_llama3_1_8b.slurm"
    exit 1
fi

export HF_TOKEN="$TOKEN_CANDIDATE"
export HUGGING_FACE_HUB_TOKEN="$TOKEN_CANDIDATE"
mkdir -p logs
mkdir -p results

echo "⏰ [$(date '+%Y-%m-%d %H:%M:%S')] Start Job $SLURM_JOB_ID on $(hostname)"

# ---------------------------------------------------------
# 1. THIẾT LẬP MÔI TRƯỜNG & VENV
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
echo "📍 [$(date '+%H:%M:%S')] Thư mục làm việc: $WORK_DIR"

export VENV_PATH="$WORK_DIR/venv_a100_8_9"

if [ ! -d "$VENV_PATH" ]; then
    echo "📦 [$(date '+%H:%M:%S')] Đang khởi tạo môi trường ảo mới tại $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip setuptools wheel --quiet
    echo "📥 Đang cài đặt PyTorch CUDA 12.1..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
    echo "📥 Đang cài đặt các thư viện yêu cầu từ kaggle/requirements.txt..."
    if [ -f "kaggle/requirements.txt" ]; then
        pip install -r kaggle/requirements.txt --quiet
    else
        pip install "transformers>=4.40.0" "accelerate>=0.28.0" "bitsandbytes>=0.43.0" "sympy>=1.12" "datasets>=2.18.0" tqdm scipy numpy --quiet
    fi
else
    echo "📦 [$(date '+%H:%M:%S')] Kích hoạt venv hiện hữu tại: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi

# Authenticate HuggingFace Hub
python3 -c "
try:
    import os
    from huggingface_hub import login
    token = os.environ.get('HF_TOKEN')
    if token:
        login(token)
        print('✅ Đăng nhập HuggingFace thành công.')
except Exception as e:
    print(f'⚠️ Lỗi đăng nhập HuggingFace: {e}')
"

# ---------------------------------------------------------
# 2. CHECK GPU CHUẨN TỰ ĐỘNG
# ---------------------------------------------------------
echo "🔍 [$(date '+%H:%M:%S')] Đang kiểm tra GPU rảnh qua gpu_check.sh..."
REQUIRED_VRAM=16384
unset CUDA_VISIBLE_DEVICES

if [ -f "/usr/local/bin/gpu_check.sh" ]; then
    CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 10 ]; then
        echo "⚠️ Hết GPU phù hợp, Slurm requeue:"
        echo "$CHECK_OUT"
        exit 0
    elif [ $EXIT_CODE -eq 11 ]; then
        echo "❌ Lỗi kiểm tra GPU: $CHECK_OUT"
        exit 1
    fi
    BEST_GPU=$CHECK_OUT
else
    BEST_GPU=0
fi

echo "✅ [$(date '+%H:%M:%S')] GPU được chọn: $BEST_GPU"

# ---------------------------------------------------------
# 3. KÍCH HOẠT MPS & CUDA VISIBLE DEVICES
# ---------------------------------------------------------
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

export CUDA_VISIBLE_DEVICES=$BEST_GPU

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

echo "🔍 [$(date '+%H:%M:%S')] Kiểm tra trạng thái GPU PyTorch:"
python3 -c "import torch; print('-> CUDA Available:', torch.cuda.is_available()); print('-> Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ---------------------------------------------------------
# 4. TRÍCH XUẤT 50 MẪU ĐÁNH GIÁ (NẾU CHƯA CÓ)
# ---------------------------------------------------------
SAMPLE_FILE="kaggle/data/math500/test_50_stratified.jsonl"

if [ ! -f "$SAMPLE_FILE" ]; then
    echo "🎲 [$(date '+%H:%M:%S')] Đang trích xuất 50 mẫu ngẫu nhiên..."
    python3 -c "
import json
import random
from collections import Counter

random.seed(42)
lvl1_3 = []
lvl4_5 = []

with open('kaggle/data/math500/test.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        lvl = item.get('level')
        if lvl in [1, 2, 3]:
            lvl1_3.append(item)
        elif lvl in [4, 5]:
            lvl4_5.append(item)

sample_35 = random.sample(lvl1_3, 35)
sample_15 = random.sample(lvl4_5, 15)
sampled_50 = sample_35 + sample_15
random.shuffle(sampled_50)

with open('$SAMPLE_FILE', 'w', encoding='utf-8') as f:
    for item in sampled_50:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f'✅ Đã lưu {len(sampled_50)} mẫu vào $SAMPLE_FILE')
"
else
    echo "📄 [$(date '+%H:%M:%S')] Đã có sẵn file 50 mẫu lưu trữ tại: $SAMPLE_FILE"
fi

# ---------------------------------------------------------
# 5. CHẠY BENCHMARK: Llama-3.1-8B-Instruct TRÊN 50 MẪU
# ---------------------------------------------------------
OUTPUT_RESULT="results/results_a100_llama3_1_8b_math500_50samples.json"

echo "=========================================================================="
echo "🚀 [$(date '+%H:%M:%S')] Bắt đầu Benchmark trên 50 mẫu MATH-500"
echo "   - Model        : meta-llama/Llama-3.1-8B-Instruct (4-bit NF4)"
echo "   - Pipelines    : Direct, CoT, SymCode, SymPlanner"
echo "   - Samples File : $SAMPLE_FILE (50 mẫu: 35 câu L1-3, 15 câu L4-5)"
echo "   - Output File  : $OUTPUT_RESULT"
echo "=========================================================================="

python kaggle/run_benchmark.py \
    --dataset math500 \
    --dataset-path "$SAMPLE_FILE" \
    --num-samples 50 \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --load-in-4bit \
    --methods Direct CoT SymCode SymPlanner \
    --run-order by-problem \
    --output-file "$OUTPUT_RESULT" \
    --timeout 15 \
    --max-retries 2 \
    --save-every 1

echo "=========================================================================="
echo "🎉 [$(date '+%H:%M:%S')] Hoàn tất Benchmark Llama-3.1-8B-Instruct!"
echo "📁 Kết quả đã được lưu tại: $OUTPUT_RESULT"
echo "=========================================================================="

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
