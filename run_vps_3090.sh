#!/bin/bash
# ==============================================================================
# Script thực thi Ablation Study trên máy VPS trang bị GPU RTX 3090 (24GB vRAM)
# Chạy trực tiếp (Không sử dụng Slurm job scheduler)
# ==============================================================================

set -e

# 1. Chuyển vào thư mục gốc của dự án
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================================================="
echo "🚀 KHỞI ĐỘNG THỰC NGHIỆM ABLATION STUDY (SymExtract vs SymPlan) TRÊN VPS 3090"
echo "=========================================================================="

# 2. Tự động khởi tạo môi trường Python venv nếu chưa tồn tại
if [ ! -d "venv" ]; then
    echo "⚠️ Tạo môi trường venv mới tại ./venv..."
    python3 -m venv venv
fi

# Kích hoạt môi trường venv
source venv/bin/activate

# 3. Cài đặt / Cập nhật 100% các thư viện phụ thuộc
echo "📦 Đang kiểm tra và cài đặt các thư viện từ kaggle/requirements.txt..."
pip install --upgrade pip
pip install -r kaggle/requirements.txt

# 4. Đặt GPU mặc định là GPU 0 (RTX 3090)
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

# Đảm bảo thư mục kết quả tồn tại
mkdir -p result

# 5. Thực thi Ablation Study trên MATH-500
echo ""
echo "🔥 [1/2] Đang chạy Ablation Study trên tập dữ liệu MATH-500..."
python kaggle/run_benchmark.py \
    --dataset math500 \
    --methods SymExtract SymPlan \
    --output-file result/math500_ablation_results.json \
    --save-every 5

# 6. Thực thi Ablation Study trên GSM8K
echo ""
echo "🔥 [2/2] Đang chạy Ablation Study trên tập dữ liệu GSM8K..."
python kaggle/run_benchmark.py \
    --dataset gsm8k \
    --methods SymExtract SymPlan \
    --output-file result/gsm8k_ablation_results.json \
    --save-every 5

echo ""
echo "=========================================================================="
echo "🎉 HOÀN TẤT TOÀN BỘ THỰC NGHIỆM ABLATION STUDY TRÊN VPS 3090!"
echo "Kết quả đã được lưu tại thư mục result/"
echo "=========================================================================="
