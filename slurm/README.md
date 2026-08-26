# Slurm Execution Scripts for Ablation Study (DGX-A100 Cluster)

Thư mục này chứa các sbatch script để chạy thực nghiệm Ablation Study trên cụm server DGX-A100 (Slurm).

## Quy tắc vận hành
1. Thư mục làm việc: `/datastore/uitchain/PNGH/symcode`
2. Đã kích hoạt MPS (`--gres=mps:2`) và giới hạn CPU `--cpus-per-task=4`.
3. Tự động kiểm tra vRAM khả dụng qua `/usr/local/bin/gpu_check.sh`.

## Các lệnh nộp Job (Run trên login node `bcm-headnode`)

```bash
# 1. Chuyển vào thư mục mã nguồn
cd /datastore/uitchain/PNGH/symcode

# 2. Nộp job chạy Ablation Study trên MATH-500
sbatch slurm/run_ablation_math500.slurm

# 3. Nộp job chạy Ablation Study trên GSM8K
sbatch slurm/run_ablation_gsm8k.slurm

# 4. Kiểm tra hàng đợi / trạng thái job
squeue -u uitchain

# 5. Xem log trực tiếp khi job đang chạy (thay <jobid> bằng ID job thực tế)
tail -f slurm/logs/ablation_math500_<jobid>.out
```
