#!/bin/bash
# ==============================================================================
# Script thực thi Ablation Study 4 luồng song song trên VPS GPU RTX 3090/4090
# (Tận dụng 100% vRAM & GPU TFLOPS)
# ==============================================================================

set -e

# 1. Chuyển vào thư mục gốc của dự án
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================================================="
echo "🚀 KHỞI ĐỘNG 4 LUỒNG SONG SONG ABLATION STUDY (SymExtract & SymPlan)"
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

# 4. Cấu hình GPU & thư mục
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

mkdir -p result logs

echo ""
echo "🔥 Đang kích hoạt 4 tiến trình Python chạy song song trên GPU 0..."

# Kích hoạt 4 luồng song song, đẩy log ra các file riêng để tránh đè tqdm
python kaggle/run_benchmark.py \
    --dataset math500 \
    --methods SymExtract \
    --output-file result/math500_extract.json \
    --save-every 5 > logs/math500_extract.log 2>&1 &

python kaggle/run_benchmark.py \
    --dataset math500 \
    --methods SymPlan \
    --output-file result/math500_plan.json \
    --save-every 5 > logs/math500_plan.log 2>&1 &

python kaggle/run_benchmark.py \
    --dataset gsm8k \
    --methods SymExtract \
    --output-file result/gsm8k_extract.json \
    --save-every 5 > logs/gsm8k_extract.log 2>&1 &

python kaggle/run_benchmark.py \
    --dataset gsm8k \
    --methods SymPlan \
    --output-file result/gsm8k_plan.json \
    --save-every 5 > logs/gsm8k_plan.log 2>&1 &

echo "✅ Cả 4 tiến trình đã được kích hoạt thành công!"
echo "📌 Xem các tiến trình Python đang chạy: ps aux | grep python"
echo "📌 Theo dõi log từng luồng: tail -f logs/math500_extract.log"
echo "=========================================================================="

# Chờ cả 4 tiến trình hoàn tất
wait
echo "🎉 HOÀN TẤT TOÀN BỘ 4 LUỒNG ABLATION STUDY!"
