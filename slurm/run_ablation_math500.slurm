#!/bin/bash
#SBATCH --job-name=ablation_math500
#SBATCH --output=logs/out_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=mps:l40:8
#SBATCH --mem=16G
#SBATCH --time=72:00:00
REQUIRED_VRAM=16384  # Quan trọng - Số vRAM cần dùng (để tìm GPU phù hợp)
# =========================================================
# CHUẨN BỊ MÔI TRƯỜNG
# =========================================================
module clear -f
module load shared python312

# *** Kích hoạt venv tự tạo ***
if [ -f "/datastore/uitchain/PNGH/venv/bin/activate" ]; then
    source /datastore/uitchain/PNGH/venv/bin/activate
elif [ -f "/datastore/uitchain/venv/bin/activate" ]; then
    source /datastore/uitchain/venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Cài đặt / Cập nhật các thư viện phụ thuộc cho môi trường mới
pip install -r kaggle/requirements.txt

# Xóa biến môi trường Slurm để tự chọn GPU
unset CUDA_VISIBLE_DEVICES
# --- GỌI HELPER --- (Quan trọng, cần gọi hàm này (có sẵn) để tìm GPU có vRAM trống >= REQUIRED_VRAM)
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0 # Thoát để Slurm đưa vào queue lại
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1 # Lỗi thật sự, dừng Job
fi
BEST_GPU=$CHECK_OUT
echo "✅ Job $SLURM_JOB_ID bắt đầu trên GPU: $BEST_GPU"
# =========================================================
# KHỞI TẠO PRIVATE MPS SERVER
# =========================================================
# Khởi tạo Private MPS với Pipe riêng cho từng JOB ID (để dễ quit)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

# Xóa sạch dấu vết cũ nếu có và tạo mới (quan trọng!)
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

export CUDA_VISIBLE_DEVICES=$BEST_GPU

export WANDB_MODE=offline

# =========================================================
# CHẠY CODE
# =========================================================
# Chỉ thực thi 2 phương pháp Ablation Study: SymExtract vs SymPlan
python kaggle/run_benchmark.py \
    --dataset math500 \
    --methods SymExtract SymPlan \
    --output-file result/math500_ablation_results.json \
    --save-every 5
