# Vast.ai: Qwen2.5-Coder-7B-Instruct 4-bit

Use a Linux NVIDIA GPU instance with a PyTorch CUDA image (the script targets CUDA 12.4).
The default run uses 50 stratified MATH-500 problems and all four methods:
Direct, CoT, SymCode, SymPlanner. Quantization is bitsandbytes NF4 with double quantization.

## First checkout

```bash
git clone --branch update-tay --single-branch https://github.com/tapu25z/symcode.git
cd symcode
mkdir -p logs
nohup bash unknow/run_vastai.sh > logs/qwen2_5_coder_7b_4bit.log 2>&1 &
echo "PID: $!"
tail -f logs/qwen2_5_coder_7b_4bit.log
```

Ctrl+C exits the log viewer; the benchmark continues in the background.
Results: `results/results_vastai_qwen2_5_coder_7b_4bit.json`.
The first run installs dependencies and downloads model weights.

## Existing checkout

With no conflicting local changes:

```bash
cd /path/to/symcode
git fetch origin
git switch update-tay
git pull --ff-only origin update-tay
```

Then run the `mkdir`, `nohup`, and `tail` commands above.

## Optional short check

Run this in the foreground before the full run if desired:

```bash
bash unknow/run_vastai.sh --num-samples 2 --output-file results/smoke_qwen2_5_coder_7b_4bit.json
```

Extra arguments override defaults without discarding the selected model or dataset.
For all 500 problems, pass `--dataset-path kaggle/data/math500/test.jsonl`.
Reusing an output filename may overwrite an earlier run; use `--output-file` to retain it.
