#!/bin/bash
#SBATCH --job-name=taybui_llama31_50pct
#SBATCH --output=logs/bench_taybui_llama31_50pct_%j.out
#SBATCH --error=logs/bench_taybui_llama31_50pct_%j.err
#SBATCH --nodelist=DGX-A100            # Chỉ định chạy trên Node DGX-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16             # 16 CPU cores
#SBATCH --gres=mps:a100:2              # Sử dụng 2 slot MPS trên GPU A100
#SBATCH --mem=32G                      # 32GB RAM hệ thống
#SBATCH --time=12:00:00                # Giới hạn thời gian chạy (12 giờ)

# ==============================================================================
# Script Slurm Benchmark: taybui-clean
# Chạy 4 Pipelines: Direct, CoT, SymCode, SymPlanner
# Tập dữ liệu: MATH-500 (50% mẫu = 250 samples)
# Model: meta-llama/Llama-3.1-8B-Instruct (4-bit NF4)
# ==============================================================================

# Tự động đẩy log ra màn hình lập tức (không bị nghẽn buffer)
export PYTHONUNBUFFERED=1

# Tạo các thư mục chứa log và kết quả
mkdir -p logs
mkdir -p taybui-clean/results

echo "⏰ [$(date '+%Y-%m-%d %H:%M:%S')] Start Job $SLURM_JOB_ID on $(hostname)"

# ---------------------------------------------------------
# 1. THIẾT LẬP MÔI TRƯỜNG & CHỌN THƯ MỤC LÀM VIỆC
# ---------------------------------------------------------
if command -v module &> /dev/null; then
    module clear -f || true
    module load shared python312 || true
fi

if [ -d "/home/uitchain/datastore/uitchain/symcode" ]; then
    export WORK_DIR="/home/uitchain/datastore/uitchain/symcode"
    export VENV_PATH="/home/uitchain/datastore/uitchain/venv_symcode"
elif [ -d "/datastore/uitchain/symcode" ]; then
    export WORK_DIR="/datastore/uitchain/symcode"
    export VENV_PATH="/datastore/uitchain/venv_symcode"
elif [ -d "$HOME/datastore/uitchain/symcode" ]; then
    export WORK_DIR="$HOME/datastore/uitchain/symcode"
    export VENV_PATH="$HOME/datastore/uitchain/venv_symcode"
else
    export WORK_DIR="$(pwd)"
    export VENV_PATH="$WORK_DIR/venv"
fi

cd "$WORK_DIR"
echo "📍 [$(date '+%H:%M:%S')] Thư mục làm việc: $WORK_DIR"

# Kích hoạt hoặc khởi tạo venv nếu chưa có
if [ ! -d "$VENV_PATH" ]; then
    echo "📦 [$(date '+%H:%M:%S')] Đang tạo venv mới tại $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip setuptools wheel --quiet
    echo "📥 Đang cài đặt PyTorch CUDA 12.1..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121
else
    source "$VENV_PATH/bin/activate"
fi

echo "📥 [$(date '+%H:%M:%S')] Kiểm tra và cập nhật các gói thư viện từ taybui-clean/requirements.txt..."
if [ -f "taybui-clean/requirements.txt" ]; then
    pip install -r taybui-clean/requirements.txt --quiet
elif [ -f "kaggle/requirements.txt" ]; then
    pip install -r kaggle/requirements.txt --quiet
else
    pip install "transformers>=4.40.0" "accelerate>=0.28.0" "bitsandbytes>=0.43.0" "sympy>=1.12" "datasets>=2.18.0" tqdm scipy numpy --quiet
fi

# ---------------------------------------------------------
# 2. CHECK GPU CHUẨN TỰ ĐỘNG (BẮT BỘC TRƯỚC KHI IMPORT TORCH)
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
# 3. KÍCH HOẠT MPS & KHỞI TẠO CUDA VISIBLE DEVICES
# ---------------------------------------------------------
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

# Khóa CUDA Visible Devices
export CUDA_VISIBLE_DEVICES=$BEST_GPU

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# ---------------------------------------------------------
# 4. KIỂM TRA PYTORCH CUDA SAU KHI ĐÃ CÓ CUDA_VISIBLE_DEVICES
# ---------------------------------------------------------
echo "🔍 [$(date '+%H:%M:%S')] Kiểm tra trạng thái GPU PyTorch:"
HAS_CUDA=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)

if [ "$HAS_CUDA" != "True" ]; then
    echo "⚠️ PyTorch chưa có CUDA! Đang tiến hành cài đặt lại PyTorch CUDA 12.1..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
    pip install "bitsandbytes>=0.43.0" "accelerate>=0.28.0" --quiet
fi

python3 -c "import torch; print('-> CUDA Available:', torch.cuda.is_available()); print('-> Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ---------------------------------------------------------
# 5. CHẠY BENCHMARK TAYBUI-CLEAN: 4 PIPELINES TRÊN 50% MATH-500
# ---------------------------------------------------------
echo "=========================================================================="
echo "🚀 [$(date '+%H:%M:%S')] Bắt đầu Benchmark 4 Pipelines trên 50% MATH-500"
echo "   - Model     : meta-llama/Llama-3.1-8B-Instruct (4-bit NF4)"
echo "   - Pipelines : Direct, CoT, SymCode, SymPlanner"
echo "   - Samples   : 250 / 500 (50%)"
echo "   - Order     : by-problem (Live comparison & per-problem checkpointing)"
echo "=========================================================================="

python taybui-clean/run_benchmark.py \
    --dataset math500 \
    --num-samples 250 \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --methods Direct CoT SymCode SymPlanner \
    --output-file taybui-clean/results/results_math500_50pct_llama3_1_8b.json \
    --timeout 15 \
    --save-every 5 \
    --run-order by-problem

echo "=========================================================================="
echo "🎉 [$(date '+%H:%M:%S')] Hoàn tất Benchmark 4 Pipelines trên 50% MATH-500!"
echo "📁 Kết quả đã được lưu tại: taybui-clean/results/results_math500_50pct_llama3_1_8b.json"
echo "=========================================================================="

# Dọn dẹp MPS pipe
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
