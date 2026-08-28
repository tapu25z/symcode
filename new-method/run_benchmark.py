"""Benchmark the old SymPlanner baseline and the new IR ablations with one model."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW_METHOD_ROOT = Path(__file__).resolve().parent
KAGGLE_ROOT = ROOT / "kaggle"
for path in (NEW_METHOD_ROOT, KAGGLE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from method import compute_metrics_table, evaluate_symplanner, execute_code_safely, load_dataset_file, save_benchmark_results
from new_method.adapters import LEGACY_7B_MODEL_ID, StageTokenBudgets, build_legacy_7b_runner
from new_method.config import ABLATIONS
from new_method.evaluator import compute_ir_diagnostics, evaluate_ir_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SymPlanner IR main-method and ablation benchmark")
    parser.add_argument("--dataset", choices=["math500", "gsm8k"], default="math500")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--methods", nargs="+", choices=["SymPlanner", *ABLATIONS], default=["SymPlanner", "IR-Codegen", "IR-BiVerify", "IR-Full"])
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--filter-levels", nargs="+", type=int, default=None)
    parser.add_argument("--model-id", default=LEGACY_7B_MODEL_ID)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
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
    dataset_path = args.dataset_path or str(KAGGLE_ROOT / "data" / args.dataset / "test.jsonl")
    output_file = args.output_file or f"{args.dataset}_symplanner_ir_ablation.json"
    dataset = load_dataset_file(dataset_path, split="test", num_samples=args.num_samples, filter_levels=args.filter_levels)
    if not dataset:
        raise SystemExit("No benchmark samples loaded")

    # This is the only call that loads the old 7B coder; importing this script does not load a model.
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
    }
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
        if method == "SymPlanner":
            benchmark_data["results"][method] = evaluate_symplanner(
                dataset, runner, timeout=args.timeout, max_retries=args.max_retries,
                checkpoint_file=output_file, save_every=args.save_every,
            )
        else:
            benchmark_data["results"][method] = evaluate_ir_variant(
                dataset, runner, execute_code_safely, variant=method,
                timeout=args.timeout, max_retries=args.max_retries,
                max_ir_retries=args.max_ir_retries, checkpoint_file=output_file,
                save_every=args.save_every, token_budgets=budgets,
            )

    benchmark_data["summary"] = compute_metrics_table(benchmark_data["results"])
    benchmark_data["ir_diagnostics"] = {
        method: compute_ir_diagnostics(benchmark_data["results"].get(method, []))
        for method in ABLATIONS if method in benchmark_data["results"]
    }
    save_benchmark_results(benchmark_data, output_file)


if __name__ == "__main__":
    main()
