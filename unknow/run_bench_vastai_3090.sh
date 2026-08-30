#!/bin/bash
# ==============================================================================
# Script thực thi Benchmark 50 mẫu MATH-500 trên VPS Vast.ai (NVIDIA RTX 3090)
# Tích hợp tự động cài đặt venv, PyTorch CUDA 12.4 và các thư viện phụ thuộc
# ==============================================================================

set -e  # Dừng ngay nếu gặp lỗi lệnh quan trọng

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "⏰ [$(date '+%Y-%m-%d %H:%M:%S')] Khởi động Script Benchmark trên Vast.ai (RTX 3090)..."

# 1. Xác định thư mục làm việc
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
echo "📍 Thư mục làm việc: $SCRIPT_DIR"

# 2. Tạo thư mục chứa log và kết quả
mkdir -p logs

# 3. Kiểm tra & Khởi tạo Virtual Environment
VENV_DIR="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 [$(date '+%H:%M:%S')] Đang tạo môi trường ảo Python (venv)..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    echo "📥 [$(date '+%H:%M:%S')] Nâng cấp pip và cài đặt PyTorch CUDA 12.4 (Cho RTX 3090)..."
    pip install --upgrade pip setuptools wheel --quiet
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    
    echo "📥 Cài đặt các gói thư viện benchmark..."
    if [ -f "kaggle/requirements.txt" ]; then
        pip install -r kaggle/requirements.txt
    else
        pip install "transformers>=4.40.0" "accelerate>=0.28.0" "bitsandbytes>=0.43.0" "sympy>=1.12" "datasets>=2.18.0" tqdm
    fi
    echo "✅ Khởi tạo venv thành công!"
else
    echo "🔄 Kích hoạt venv sẵn có tại: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

# 4. Kiểm tra trạng thái GPU & PyTorch CUDA 12.4
echo "🔍 [$(date '+%H:%M:%S')] Kiểm tra trạng thái GPU với nvidia-smi:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

echo "🔍 Kiểm tra nhận diện GPU qua PyTorch:"
HAS_CUDA=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "$HAS_CUDA" != "True" ]; then
    echo "⚠️ PyTorch chưa nhận CUDA! Đang tiến hành cài đặt lại PyTorch CUDA 12.4..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
    pip install "bitsandbytes>=0.43.0" "accelerate>=0.28.0" --quiet
fi

python3 -c "import torch; print('-> CUDA Available:', torch.cuda.is_available()); print('-> Device Count:', torch.cuda.device_count()); print('-> Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 5. Chạy Benchmark 50 mẫu MATH-500 với phương pháp SymPlanner
echo "========================================================="
echo "🚀 [$(date '+%H:%M:%S')] Bắt đầu Benchmark 50 mẫu MATH-500 trên RTX 3090..."
echo "========================================================="

python kaggle/run_benchmark.py \
    --dataset math500 \
    --num-samples 50 \
    --methods SymPlanner \
    --output-file result_symplanner_math500_n50_vastai.json \
    --save-every 5 2>&1 | tee logs/bench_vastai_3090.log

echo "🎉 [$(date '+%H:%M:%S')] Hoàn tất Benchmark 50 mẫu MATH-500 trên Vast.ai!"
