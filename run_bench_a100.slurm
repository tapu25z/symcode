#!/bin/bash
#SBATCH --job-name=symplanner_a100
#SBATCH --output=logs/bench_a100_%j.out
#SBATCH --error=logs/bench_a100_%j.err
#SBATCH --nodelist=DGX-A100            # Chỉ định chạy trực tiếp trên Node DGX-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16             # Đòi hỏi 16 CPU cores
#SBATCH --gres=mps:a100:2              # Sử dụng 2 slot MPS trên GPU A100
#SBATCH --mem=32G                      # 32GB RAM hệ thống
#SBATCH --time=12:00:00                # Thời gian chạy tối đa

# Tự động đẩy log ra màn hình lập tức (không bị nghẽn buffer)
export PYTHONUNBUFFERED=1

# Tạo thư mục chứa log
mkdir -p logs

echo "⏰ [$(date '+%Y-%m-%d %H:%M:%S')] Start Job $SLURM_JOB_ID"

# ---------------------------------------------------------
# 1. THIẾT LẬP MÔI TRƯỜNG & CHỌN THƯ MỤC
# ---------------------------------------------------------
module clear -f
module load shared python312

if [ -d "/home/uitchain/datastore/uitchain/symcode" ]; then
    export WORK_DIR="/home/uitchain/datastore/uitchain/symcode"
    export VENV_PATH="/home/uitchain/datastore/uitchain/venv_symcode"
elif [ -d "/datastore/uitchain/symcode" ]; then
    export WORK_DIR="/datastore/uitchain/symcode"
    export VENV_PATH="/datastore/uitchain/venv_symcode"
else
    export WORK_DIR="$HOME/datastore/uitchain/symcode"
    export VENV_PATH="$HOME/datastore/uitchain/venv_symcode"
fi

cd $WORK_DIR
echo "📍 [$(date '+%H:%M:%S')] Thư mục làm việc: $WORK_DIR"

if [ ! -d "$VENV_PATH" ]; then
    echo "📦 [$(date '+%H:%M:%S')] Đang tạo venv mới tại $VENV_PATH..."
    python3 -m venv $VENV_PATH
    source $VENV_PATH/bin/activate
    pip install --upgrade pip setuptools wheel --quiet
    echo "📥 Đang tải PyTorch CUDA 12.1..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    if [ -f "kaggle/requirements.txt" ]; then
        pip install -r kaggle/requirements.txt
    else
        pip install "transformers>=4.40.0" "accelerate>=0.28.0" "bitsandbytes>=0.43.0" "sympy>=1.12" "datasets>=2.18.0" tqdm
    fi
else
    source $VENV_PATH/bin/activate
fi

# ---------------------------------------------------------
# 2. CHECK GPU CHUẨN TỰ ĐỘNG (BẮT BỘC CHẠY TRƯỚC KHI IMPORT TORCH)
# ---------------------------------------------------------
echo "🔍 [$(date '+%H:%M:%S')] Đang kiểm tra GPU rảnh qua gpu_check.sh..."
REQUIRED_VRAM=16384
unset CUDA_VISIBLE_DEVICES

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
echo "✅ [$(date '+%H:%M:%S')] GPU được chọn: $BEST_GPU"

# ---------------------------------------------------------
# 3. KÍCH HOẠT MPS & KHỞI TẠO CUDA VISIBLE DEVICES
# ---------------------------------------------------------
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

# KHÓA CUDA VISIBLE DEVICES ĐỂ PHÒNG TREO IMPORT TORCH
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
# 5. CHẠY BENCHMARK PHƯƠNG PHÁP MỚI (SymPlanner)
# ---------------------------------------------------------
echo "========================================================="
echo "🚀 [$(date '+%H:%M:%S')] [1/2] Bắt đầu Benchmark SymPlanner trên MATH-500..."
echo "========================================================="
python kaggle/run_benchmark.py \
    --dataset math500 \
    --methods SymPlanner \
    --output-file result_symplanner_math500.json \
    --timeout 15 \
    --save-every 5

echo "========================================================="
echo "🚀 [$(date '+%H:%M:%S')] [2/2] Bắt đầu Benchmark SymPlanner trên GSM8K..."
echo "========================================================="
python kaggle/run_benchmark.py \
    --dataset gsm8k \
    --methods SymPlanner \
    --output-file result_symplanner_gsm8k.json \
    --timeout 15 \
    --save-every 10

echo "🎉 [$(date '+%H:%M:%S')] Hoàn tất Benchmark cả 2 dataset!"

# Dọn dẹp MPS pipe
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
