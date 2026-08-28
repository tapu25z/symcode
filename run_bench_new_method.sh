#!/bin/bash
# ==============================================================================
# Script thực thi benchmark nhánh new-method (SymPlanner IR)
# Chạy so sánh đối đầu: baseline SymPlanner vs pipeline IR duy nhất
# ==============================================================================

set -e

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "⏰ [$(date '+%Y-%m-%d %H:%M:%S')] Khởi động Benchmark new-method (SymPlanner IR)..."
echo "📍 Thư mục làm việc: $SCRIPT_DIR"

# 1. Kích hoạt môi trường venv
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Đang tạo venv mới..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel --quiet
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install -r kaggle/requirements.txt
else
    echo "🔄 Kích hoạt venv tại $VENV_DIR..."
    source "$VENV_DIR/bin/activate"
fi

# 2. Cấu hình PYTHONPATH để nạp các module
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/new-method:$SCRIPT_DIR/kaggle:$PYTHONPATH"

# 3. Tạo thư mục chứa log
mkdir -p logs

# 4. Thực thi benchmark new-method
echo "========================================================="
echo "🚀 Bắt đầu Benchmark 50 mẫu trên nhánh new-method..."
echo "========================================================="

python new-method/run_benchmark.py \
    --dataset math500 \
    --num-samples 50 \
    --methods SymPlanner IR \
    --output-file math500_symplanner_ir_ablation_n50.json \
    --save-every 5 2>&1 | tee logs/bench_new_method.log

echo "🎉 [$(date '+%H:%M:%S')] Hoàn tất Benchmark new-method!"
