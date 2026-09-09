#!/usr/bin/env bash
# ==============================================================================
# Script tu dong chay toan bo 8 Models Benchmark Full MATH-500
# Che do: Sequential (Tuan tu tung model tren 1 GPU RTX 3090 / 4090, CUDA 12.4)
# Thu tu: NGUOC TU DUOI LEN TREN (Reverse Order)
# ==============================================================================
# Usage:
#   chmod +x unknow/run_all_sequential_reverse.sh
#   ./unknow/run_all_sequential_reverse.sh                 # Chay tu dong
#   ./unknow/run_all_sequential_reverse.sh <HF_TOKEN>      # Truyen truc tiep HF_TOKEN
# ==============================================================================

# ==============================================================================
# 0. HUGGING FACE TOKEN (Dien truc tiep key hf_... vao day neu muon)
# ==============================================================================
DEFAULT_HF_TOKEN=""

TOKEN_CANDIDATE="${1:-${HF_TOKEN:-$DEFAULT_HF_TOKEN}}"

if [ -z "$TOKEN_CANDIDATE" ]; then
    if [ -f "$HOME/.cache/huggingface/token" ]; then
        TOKEN_CANDIDATE=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null | tr -d '\r\n')
    elif [ -f ".hf_token" ]; then
        TOKEN_CANDIDATE=$(cat ".hf_token" 2>/dev/null | tr -d '\r\n')
    elif [ -f "$HOME/.hf_token" ]; then
        TOKEN_CANDIDATE=$(cat "$HOME/.hf_token" 2>/dev/null | tr -d '\r\n')
    fi
fi

if [ -n "$TOKEN_CANDIDATE" ]; then
    export HF_TOKEN="$TOKEN_CANDIDATE"
    export HUGGING_FACE_HUB_TOKEN="$TOKEN_CANDIDATE"
fi

# ---------------------------------------------------------
# 1. CAU HINH HE THONG & GPU TREN VAST.AI
# ---------------------------------------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p results logs

echo "=============================================================================="
echo "🚀 KHOI CHAY CHUOI BENCHMARK TUAN TU (SEQUENTIAL REVERSE ORDER)"
echo "   - Thiet bi       : GPU ${CUDA_VISIBLE_DEVICES} (RTX 3090 / 4090, CUDA 12.4)"
echo "   - Tap du lieu    : MATH-500 (Full 500 mau)"
echo "   - Pipelines      : Direct, CoT, SymCode, SymPlanner"
echo "   - Huong chay     : Nguoc tu duoi len tren (8 models)"
echo "=============================================================================="

# ---------------------------------------------------------
# 2. KIEM TRA GPU & DRIVER
# ---------------------------------------------------------
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 [$(date '+%H:%M:%S')] Thong tin GPU:"
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
else
    echo "⚠️ [$(date '+%H:%M:%S')] Khong tim thay nvidia-smi. Hay dam bao driver da duoc nap."
fi

# ---------------------------------------------------------
# 3. KIEM TRA / CAI DAT PYTORCH CUDA 12.4 & THU VIEN
# ---------------------------------------------------------
echo "🔍 [$(date '+%H:%M:%S')] Kiem tra PyTorch va CUDA..."
python3 -c "import torch; assert torch.cuda.is_available(); print(f'   -> PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || {
    echo "📥 [$(date '+%H:%M:%S')] Dang cai dat / cap nhat PyTorch CUDA 12.4 cho RTX 3090 / 4090..."
    pip install --upgrade pip --quiet
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
}

if [ -f "kaggle/requirements.txt" ]; then
    echo "📥 [$(date '+%H:%M:%S')] Dong bo thu vien tu kaggle/requirements.txt..."
    pip install -q -r kaggle/requirements.txt
else
    pip install "transformers>=4.40.0" "accelerate>=0.28.0" "bitsandbytes>=0.43.0" "sympy>=1.12" "datasets>=2.18.0" tqdm scipy numpy --quiet
fi

# ---------------------------------------------------------
# 4. DANG NHAP HUGGING FACE TU DONG
# ---------------------------------------------------------
if [ -n "$HF_TOKEN" ]; then
    python3 -c "
import os
try:
    from huggingface_hub import login
    login(os.environ['HF_TOKEN'])
    print('✅ Dang nhap Hugging Face thanh cong!')
except Exception as e:
    print(f'⚠️ Loi dang nhap Hugging Face: {e}')
"
else
    echo "ℹ️ [INFO] Khong tim thay HF_TOKEN. Cac model open (Qwen, DeepSeek) van tai binh thuong."
fi

# ---------------------------------------------------------
# 5. DANH SACH 8 MODELS THEO THU TU NGUOC TU DUOI LEN TREN
# ---------------------------------------------------------
MODELS=(
    "meta-llama/Llama-3.2-3B-Instruct|results/results_full_math500_llama3_2_3b.json|Llama-3.2-3B"
    "Qwen/Qwen2.5-3B-Instruct|results/results_full_math500_qwen2_5_3b.json|Qwen2.5-3B"
    "Qwen/Qwen2.5-7B-Instruct|results/results_full_math500_qwen2_5_7b.json|Qwen2.5-7B"
    "Qwen/Qwen3-8B|results/results_full_math500_qwen3_8b.json|Qwen3-8B"
    "deepseek-ai/deepseek-coder-6.7b-instruct|results/results_full_math500_deepseek_coder_6_7b.json|DeepSeek-Coder-6.7B"
    "Qwen/Qwen2.5-Coder-7B-Instruct|results/results_full_math500_qwen2_5_coder_7b.json|Qwen2.5-Coder-7B"
    "meta-llama/Llama-3.1-8B-Instruct|results/results_full_math500_llama3_1_8b.json|Llama-3.1-8B"
    "Qwen/Qwen3-4B|results/results_full_math500_qwen3_4b.json|Qwen3-4B"
)

TOTAL_MODELS=${#MODELS[@]}
echo ""
echo "📋 Danh sach 8 model se chay tuan tu tu duoi len tren:"
for i in "${!MODELS[@]}"; do
    IFS='|' read -r m_id m_out m_name <<< "${MODELS[$i]}"
    echo "   $((i+1)). $m_name ($m_id)"
done
echo "=============================================================================="

# ---------------------------------------------------------
# 6. VONG LAP CHAY TUAN TU TUNG MODEL
# ---------------------------------------------------------
START_ALL_TIME=$(date +%s)

for idx in "${!MODELS[@]}"; do
    IFS='|' read -r MODEL_ID OUTPUT_RESULT DISPLAY_NAME <<< "${MODELS[$idx]}"
    STEP=$((idx + 1))
    
    echo ""
    echo "##############################################################################"
    echo "▶️  [MODEL $STEP/$TOTAL_MODELS] BAT DAU BENCHMARK: $DISPLAY_NAME"
    echo "   - Model ID    : $MODEL_ID"
    echo "   - Output File : $OUTPUT_RESULT"
    echo "   - Thoi gian   : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "💾 Dung luong o dia truoc khi chay:"
    df -h / | awk 'NR==1 || NR==2'
    echo "##############################################################################"

    # Kiem tra neu model da hoan thanh du 500 mau thi bo qua
    if [ -f "$OUTPUT_RESULT" ]; then
        COMPLETED_COUNT=$(python3 -c "
import json
try:
    with open('$OUTPUT_RESULT') as f:
        d = json.load(f)
    counts = [len(v) for v in d.get('results', {}).values()]
    print(min(counts) if counts else 0)
except Exception:
    print(0)
" 2>/dev/null || echo 0)

        if [ "$COMPLETED_COUNT" -ge 500 ]; then
            echo "✅ [SKIP] Model $DISPLAY_NAME da hoan thanh du 500/500 mau! Chuyen sang model tiep theo."
            continue
        elif [ "$COMPLETED_COUNT" -gt 0 ]; then
            echo "🔄 [RESUME] Model $DISPLAY_NAME da hoan thanh $COMPLETED_COUNT/500 mau. Dang chay tiep..."
        fi
    fi

    LOG_FILE="logs/seq_${idx}_$(echo "$DISPLAY_NAME" | tr '[:upper:]' '[:lower:]' | tr '-' '_')_$(date '+%Y%m%d_%H%M%S').log"
    echo "📁 Log file: $LOG_FILE"

    # Chay benchmark va pipe dong thoi ra terminal + log file
    python3 kaggle/run_benchmark.py \
        --dataset math500 \
        --model-id "$MODEL_ID" \
        --load-in-4bit \
        --methods Direct CoT SymCode SymPlanner \
        --run-order by-problem \
        --output-file "$OUTPUT_RESULT" \
        --timeout 15 \
        --max-retries 2 \
        --save-every 1 2>&1 | tee -a "$LOG_FILE" || {
            echo "❌ [ERROR] Model $DISPLAY_NAME gap loi khi chay! Ghi nhan loi va chuyen sang model tiep theo."
        }

    # ---------------------------------------------------------
    # GIAI PHONG VRAM GPU
    # ---------------------------------------------------------
    echo "🧹 [$(date '+%H:%M:%S')] Don dep VRAM GPU truoc khi sang model tiep theo..."
    python3 -c "import torch, gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true

    # ---------------------------------------------------------
    # XOA CACHE WEIGHTS CUA MODEL DE TIET KIEM DUNG LUONG (DISK <= 50GB)
    # ---------------------------------------------------------
    HF_HUB_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    MODEL_DIR_NAME="models--$(echo "$MODEL_ID" | sed 's/\//--/g')"
    TARGET_CACHE="$HF_HUB_DIR/$MODEL_DIR_NAME"

    if [ -d "$TARGET_CACHE" ]; then
        SIZE_FREED=$(du -sh "$TARGET_CACHE" 2>/dev/null | cut -f1 || echo "N/A")
        echo "🗑️ [$(date '+%H:%M:%S')] Dang xoa cache weights cua $MODEL_ID (giai phong ~$SIZE_FREED disk)..."
        rm -rf "$TARGET_CACHE"
        echo "✅ Da xoa thanh cong cache cua $MODEL_ID!"
    fi

    # Don dep them cac file lock va temp cua transformers/huggingface
    rm -rf "$HF_HUB_DIR/.locks" 2>/dev/null || true
    rm -rf /tmp/transformers* /tmp/huggingface* 2>/dev/null || true

    # Hien thi dung luong o dia con lai tren instance sau khi xoa
    echo "💾 [$(date '+%H:%M:%S')] Dung luong o dia con lai tren instance sau khi don dep:"
    df -h / | awk 'NR==1 || NR==2'
    sleep 3

    echo "✅ [HOAN TAT $STEP/$TOTAL_MODELS] $DISPLAY_NAME"
done

# ---------------------------------------------------------
# 7. TONG KET KET QUA BENCHMARK TOAN BO 8 MODELS
# ---------------------------------------------------------
END_ALL_TIME=$(date +%s)
TOTAL_SECS=$((END_ALL_TIME - START_ALL_TIME))

echo ""
echo "=============================================================================="
echo "🎉 DA HOAN TAT TOAN BO CHUOI BENCHMARK 8 MODELS!"
echo "⏱️  Tong thoi gian chay: $(date -u -d @"$TOTAL_SECS" +'%H:%M:%S') ($TOTAL_SECS giay)"
echo "=============================================================================="
echo ""
echo "📊 BẢNG TỔNG KẾT TỈ LỆ ĐÚNG TRÊN MATH-500:"
printf "%-26s | %-10s | %-10s | %-10s | %-10s\n" "Model" "Direct" "CoT" "SymCode" "SymPlanner"
echo "---------------------------+------------+------------+------------+------------"

for idx in "${!MODELS[@]}"; do
    IFS='|' read -r MODEL_ID OUTPUT_RESULT DISPLAY_NAME <<< "${MODELS[$idx]}"
    if [ -f "$OUTPUT_RESULT" ]; then
        python3 -c "
import json
try:
    with open('$OUTPUT_RESULT') as f:
        d = json.load(f)
    res = d.get('results', {})
    def acc(m):
        items = res.get(m, [])
        if not items: return 'N/A'
        c = sum(1 for x in items if x.get('is_correct'))
        return f'{c/len(items)*100:5.2f}%'
    print(f'${DISPLAY_NAME:<26} | {acc(\"Direct\"):<10} | {acc(\"CoT\"):<10} | {acc(\"SymCode\"):<10} | {acc(\"SymPlanner\"):<10}')
except Exception as e:
    print(f'${DISPLAY_NAME:<26} | Error reading result')
"
    else
        printf "%-26s | Chưa có file kết quả\n" "$DISPLAY_NAME"
    fi
done

echo "=============================================================================="
