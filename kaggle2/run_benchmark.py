"""
Script thuc thi Benchmark qua giao dien dong lenh (CLI):
- Baselines: Direct (methods/direct), CoT (methods/cot), SymCode (methods/symcode)
- Ours: SymPlan (methods/symplan)
- Ablations: IR-Codegen, IR-BiVerify
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE2_ROOT = Path(__file__).resolve().parent
for path in (KAGGLE2_ROOT, KAGGLE2_ROOT / "shared", KAGGLE2_ROOT / "methods", ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared import (
    load_dataset_file,
    compute_metrics_table,
    save_benchmark_results,
    execute_code_safely
)
from methods.direct import evaluate_direct
from methods.cot import evaluate_cot
from methods.symcode import evaluate_symcode
from methods.symplan import (
    LEGACY_7B_MODEL_ID,
    StageTokenBudgets,
    build_legacy_7b_runner,
    evaluate_symplan,
    evaluate_ir_variant,
    compute_ir_diagnostics,
    ABLATIONS
)

ALL_SUPPORTED_METHODS = ["Direct", "CoT", "SymCode", "SymPlan", "IR-Full", "IR-Codegen", "IR-BiVerify"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Reasoning Benchmark Suite (Baselines: Direct, CoT, SymCode | Ours: SymPlan)")
    parser.add_argument("--dataset", choices=["math500", "gsm8k"], default="math500", help="Dataset: 'math500' hoac 'gsm8k'.")
    parser.add_argument("--dataset-path", default=None, help="Duong dan file JSONL.")
    parser.add_argument("--methods", nargs="+", choices=ALL_SUPPORTED_METHODS, default=["Direct", "CoT", "SymCode", "SymPlan"], help="Danh sach cac phuong phap can danh gia.")
    parser.add_argument("--num-samples", type=int, default=None, help="So luong mau danh gia (None = toan bo).")
    parser.add_argument("--filter-levels", nargs="+", type=int, default=None, help="Loc muc do kho (vi du: --filter-levels 1 2 3).")
    parser.add_argument("--model-id", default=LEGACY_7B_MODEL_ID, help="Hugging Face Model ID.")
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="Su dung luong tu hoa 4-bit NF4.")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false", help="Tat 4-bit, su dung float16/bfloat16.")
    parser.add_argument("--max-input-tokens", type=int, default=6144, help="Max input context length.")
    parser.add_argument("--max-new-tokens", type=int, default=1800, help="Max new tokens per generation.")
    parser.add_argument("--extractor-tokens", type=int, default=1400)
    parser.add_argument("--ir-repair-tokens", type=int, default=1400)
    parser.add_argument("--codegen-tokens", type=int, default=1800)
    parser.add_argument("--code-repair-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-ir-retries", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--output-file", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.dataset_path:
        dataset_path = args.dataset_path
    elif (KAGGLE2_ROOT / "data" / args.dataset / "test.jsonl").exists():
        dataset_path = str(KAGGLE2_ROOT / "data" / args.dataset / "test.jsonl")
    else:
        dataset_path = str(ROOT / "kaggle" / "data" / args.dataset / "test.jsonl")

    if args.output_file:
        output_file = args.output_file
    else:
        lvl_str = ("_lvl" + "_".join(str(lvl) for lvl in sorted(args.filter_levels))) if args.filter_levels else ""
        output_file = f"{args.dataset}{lvl_str}_benchmark_results.json"

    dataset = load_dataset_file(dataset_path, split="test", num_samples=args.num_samples, filter_levels=args.filter_levels)
    if not dataset:
        raise SystemExit(f"Khong tim thay du lieu danh gia tai {dataset_path}")

    runner = build_legacy_7b_runner(
        model_id=args.model_id,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        temperature=args.temperature,
    )
    budgets = StageTokenBudgets(args.extractor_tokens, args.ir_repair_tokens, args.codegen_tokens, args.code_repair_tokens)

    config = {
        "model_id": args.model_id,
        "dataset_name": args.dataset,
        "dataset_path": dataset_path,
        "methods_to_run": args.methods,
        "num_samples": args.num_samples,
        "filter_levels": args.filter_levels,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "max_retries": args.max_retries,
        "max_ir_retries": args.max_ir_retries,
        "timeout": args.timeout,
        "stage_token_budgets": budgets.__dict__,
        "output_file": output_file
    }

    print("=" * 75)
    print(f"[INFO] Mo hinh: {config['model_id']}")
    print(f"[INFO] Tap du lieu: {config['dataset_name'].upper()} ({config['dataset_path']})")
    print(f"[INFO] Loc do kho: {config['filter_levels']}")
    print(f"[INFO] So mau: {config['num_samples'] if config['num_samples'] is not None else 'TOAN BO'}")
    print(f"[INFO] Cac phuong phap: {config['methods_to_run']}")
    print(f"[INFO] File ket qua: {config['output_file']}")
    print("=" * 75)

    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as handle:
                benchmark_data = json.load(handle)
        except Exception:
            benchmark_data = {}
    else:
        benchmark_data = {}

    benchmark_data.update({"config": config, "timestamp": benchmark_data.get("timestamp") or time.strftime("%Y-%m-%d %H:%M:%S")})
    benchmark_data.setdefault("results", {})

    for method in args.methods:
        print("\n" + "=" * 75)
        print(f"[RUNNING METHOD] ===> {method}")
        print("=" * 75)

        if method == "Direct":
            benchmark_data["results"]["Direct"] = evaluate_direct(
                dataset, runner,
                checkpoint_file=output_file, save_every=args.save_every
            )
        elif method == "CoT":
            benchmark_data["results"]["CoT"] = evaluate_cot(
                dataset, runner,
                checkpoint_file=output_file, save_every=args.save_every
            )
        elif method == "SymCode":
            benchmark_data["results"]["SymCode"] = evaluate_symcode(
                dataset, runner,
                timeout=args.timeout, max_retries=args.max_retries,
                checkpoint_file=output_file, save_every=args.save_every
            )
        elif method in ["SymPlan", "IR-Full"]:
            benchmark_data["results"][method] = evaluate_symplan(
                dataset, runner, execute_code_safely,
                variant="SymPlan", timeout=args.timeout, max_retries=args.max_retries,
                max_ir_retries=args.max_ir_retries, checkpoint_file=output_file,
                save_every=args.save_every, token_budgets=budgets
            )
        elif method in ABLATIONS:
            benchmark_data["results"][method] = evaluate_ir_variant(
                dataset, runner, execute_code_safely,
                variant=method, timeout=args.timeout, max_retries=args.max_retries,
                max_ir_retries=args.max_ir_retries, checkpoint_file=output_file,
                save_every=args.save_every, token_budgets=budgets
            )
        else:
            print(f"[WARN] Khong nhan dien phuong phap: {method}. Bo qua...")

    benchmark_data["summary"] = compute_metrics_table(benchmark_data["results"])
    benchmark_data["ir_diagnostics"] = {
        method: compute_ir_diagnostics(benchmark_data["results"].get(method, []))
        for method in list(ABLATIONS.keys()) + ["SymPlan", "IR-Full"]
        if method in benchmark_data["results"]
    }
    save_benchmark_results(benchmark_data, output_file)
    print(f"\n[SUCCESS] Hoan thanh toan bo danh gia! File ket qua: {output_file}")


if __name__ == "__main__":
    main()
