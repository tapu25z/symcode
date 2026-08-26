#!/bin/bash
#SBATCH --job-name=ablation_gsm8k
#SBATCH --chdir=/datastore/uitchain/PNGH/symcode
#SBATCH --output=slurm/logs/ablation_gsm8k_%j.out
#SBATCH --error=slurm/logs/ablation_gsm8k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=mps:2
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# vRAM cần dùng cho Qwen2.5-Coder-7B 4-bit (MB)
REQUIRED_VRAM=16384

# =========================================================
# 1. CHUẨN BỊ MÔI TRƯỜNG SLURM & PYTHON
# =========================================================
module clear -f
module load shared python312

# Kích hoạt venv (Cấu hình theo đường dẫn /datastore/uitchain/...)
if [ -d "/datastore/uitchain/venv" ]; then
    source /datastore/uitchain/venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

unset CUDA_VISIBLE_DEVICES

# =========================================================
# 2. HELPER CHỌN GPU TỰ ĐỘNG THEO VRAM TRỐNG
# =========================================================
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 10 ]; then
    echo "⚠️ Hết GPU thỏa mãn vRAM >= ${REQUIRED_VRAM}MB. Trả job lại hàng đợi Slurm để chờ."
    echo "$CHECK_OUT"
    exit 0
elif [ $EXIT_CODE -eq 11 ]; then
    echo "❌ Lỗi hệ thống khi kiểm tra GPU:"
    echo "$CHECK_OUT"
    exit 1
fi

BEST_GPU=$CHECK_OUT
echo "✅ Job $SLURM_JOB_ID bắt đầu trên GPU: $BEST_GPU"

# =========================================================
# 3. KHỞI TẠO PRIVATE MPS SERVER DÀNH RIÊNG CHO JOB
# =========================================================
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU

# =========================================================
# 4. THỰC THI CHẠY BENCHMARK ABLATION STUDY
# =========================================================
# Chạy cả 6 phương pháp trên GSM8K
python kaggle/run_benchmark.py \
    --dataset gsm8k \
    --methods Direct CoT SymCode SymExtract SymPlan SymPlanner \
    --output-file result/gsm8k_ablation_results.json \
    --save-every 5

echo "🎉 Job $SLURM_JOB_ID hoàn tất thực nghiệm Ablation trên GSM8K!"
