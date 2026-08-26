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
# CHUẨN BỊ MÔI TRƯỜNG (TỰ ĐỘNG TẠO MỚI VENV NẾU CHƯA CÓ)
# =========================================================
module clear -f
module load shared python312

# Đường dẫn venv tự tạo của user trên /datastore
VENV_DIR="/datastore/uitchain/PNGH/venv"

# Tự động khởi tạo môi trường venv mới hoàn toàn nếu chưa tồn tại
if [ ! -d "$VENV_DIR" ]; then
    echo "⚠️ Môi trường venv chưa tồn tại tại $VENV_DIR. Đang tự động khởi tạo mới..."
    python3 -m venv "$VENV_DIR"
fi

# Kích hoạt venv
source "$VENV_DIR/bin/activate"

# Cài đặt / Cập nhật 100% các thư viện phụ thuộc để tránh thiếu thư viện
pip install --upgrade pip
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
