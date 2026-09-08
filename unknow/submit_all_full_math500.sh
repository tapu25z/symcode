#!/usr/bin/env bash
# ==============================================================================
# Script tu dong submit toan bo 8 bai toan Benchmark FULL MATH-500 len Slurm
# ==============================================================================
# Usage:
#   bash unknow/submit_all_full_math500.sh
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Dang submit 8 jobs Slurm Benchmark Full MATH-500..."

sbatch "$DIR/run_full_qwen3_4b.slurm"
sbatch "$DIR/run_full_llama3_1_8b.slurm"
sbatch "$DIR/run_full_qwen2_5_coder_7b.slurm"
sbatch "$DIR/run_full_deepseek_coder_6_7b.slurm"
sbatch "$DIR/run_full_qwen3_8b.slurm"
sbatch "$DIR/run_full_qwen2_5_7b.slurm"
sbatch "$DIR/run_full_qwen2_5_3b.slurm"
sbatch "$DIR/run_full_llama3_2_3b.slurm"

echo "✅ Da submit thanh cong ca 8 jobs len Slurm!"
echo "Kiem tra tien do bang: squeue -u $(whoami)"
