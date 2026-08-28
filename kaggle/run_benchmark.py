"""
Script thuc thi Benchmark danh gia nang luc suy luan toan hoc (Direct, CoT, SymCode).
Ho tro chay truc tiep qua Command Line Interface (CLI) voi day du cac tuy chon tham so.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Optional

from method import (
    LLMRunner,
    load_dataset_file,
    evaluate_direct_or_cot,
    evaluate_symcode,
    evaluate_symplanner,
    compute_metrics_table,
    save_benchmark_results
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Reasoning Benchmark Suite (Direct, CoT, SymCode, SymPlanner)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        choices=["math500", "gsm8k"],
        help="Tap du lieu benchmark: 'math500' hoac 'gsm8k'."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Duong dan file JSONL cuc bo. Neu bo trong, tu dong xac dinh theo thu muc data/."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["Direct", "CoT", "SymCode", "SymPlanner"],
        choices=["Direct", "CoT", "SymCode", "SymPlanner"],
        help="Danh sach cac phuong phap can danh gia."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="So luong mau toi da can danh gia (None = chay toan bo)."
    )
    parser.add_argument(
        "--filter-levels",
        nargs="+",
        type=int,
        default=None,
        help="Loc danh sach cac muc do kho (vi du: --filter-levels 1 2 3)."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="ID mo hinh tren Hugging Face."
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="Su dung luong tu hoa 4-bit NF4 via bitsandbytes."
    )
    parser.add_argument(
        "--no-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Tat luong tu hoa 4-bit, su dung float16/bfloat16 day du."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="So token toi da cho moi lan sinh."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Nhiet do giai ma (0.0 = Greedy Search)."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="So lan thu lai toi da cho vong lap tu sua loi SymCode."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3,
        help="Thoi gian timeout thuc thi code sandbox (tinh bang giay)."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Duong dan file JSON luu ket qua."
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        default=False,
        help="Lay N mau CUOI CUNG thay vi N mau dau tien."
    )
    parser.add_argument(
        "--per-level-samples",
        type=int,
        default=None,
        help="So luong mau lay cho MOI LEVEL (vi du: --per-level-samples 100)."
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Tan suat luu checkpoint trung gian theo so mau."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Xac dinh duong dan file du lieu
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        if args.dataset == "math500":
            dataset_path = os.path.join(os.path.dirname(__file__), "data", "math500", "test.jsonl")
        else:
            dataset_path = os.path.join(os.path.dirname(__file__), "data", "gsm8k", "test.jsonl")

    # Xac dinh duong dan file output
    if args.output_file:
        output_file = args.output_file
    else:
        if args.filter_levels and len(args.filter_levels) > 0:
            lvl_str = "_lvl" + "_".join(str(l) for l in sorted(args.filter_levels))
        else:
            lvl_str = ""
        tail_str = "_tail" if args.tail else ""
        output_file = f"{args.dataset}{lvl_str}{tail_str}_results.json"

    config = {
        "model_id": args.model_id,
        "load_in_4bit": args.load_in_4bit,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "dataset_name": args.dataset,
        "dataset_path": dataset_path,
        "filter_levels": args.filter_levels,
        "num_samples": args.num_samples,
        "per_level_samples": args.per_level_samples,
        "tail": args.tail,
        "methods_to_run": args.methods,
        "code_exec_timeout": args.timeout,
        "max_symcode_retries": args.max_retries,
        "output_file": output_file,
        "save_every": args.save_every
    }

    print("=" * 75)
    print(f"[INFO] Mo hinh: {config['model_id']}")
    print(f"[INFO] Tap du lieu: {config['dataset_name'].upper()} ({config['dataset_path']})")
    print(f"[INFO] Loc do kho: {config['filter_levels']}")
    print(f"[INFO] Mau moi level: {config['per_level_samples']}")
    print(f"[INFO] Lat nguoc lay tu cuoi (tail): {config['tail']}")
    print(f"[INFO] So mau danh gia: {config['num_samples'] if config['num_samples'] is not None else 'TOAN BO'}")
    print(f"[INFO] Cac phuong phap: {config['methods_to_run']}")
    print(f"[INFO] File ket qua: {config['output_file']}")
    print("=" * 75)

    # Nap dataset
    dataset = load_dataset_file(
        dataset_path,
        split="test",
        num_samples=args.num_samples,
        filter_levels=args.filter_levels,
        tail=args.tail,
        per_level_samples=args.per_level_samples
    )


    if not dataset:
        print("[ERROR] Khong tim thay du lieu de danh gia. Kiem tra lai duong dan file.")
        sys.exit(1)

    # Khoi tao mo hinh
    llm = LLMRunner(
        model_id=args.model_id,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    # Doc checkpoint da co neu ton tai
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                benchmark_data = json.load(f)
        except Exception:
            benchmark_data = {"config": config, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": {}, "summary": {}}
    else:
        benchmark_data = {"config": config, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": {}, "summary": {}}

    # Thuc thi tung phuong phap
    for method in args.methods:
        if method in ["Direct", "CoT"]:
            benchmark_data["results"][method] = evaluate_direct_or_cot(
                method, dataset, llm, checkpoint_file=output_file, save_every=args.save_every
            )
        elif method == "SymCode":
            benchmark_data["results"]["SymCode"] = evaluate_symcode(
                dataset, llm, timeout=args.timeout, max_retries=args.max_retries, checkpoint_file=output_file, save_every=args.save_every
            )
        elif method == "SymPlanner":
            benchmark_data["results"]["SymPlanner"] = evaluate_symplanner(
                dataset, llm, timeout=args.timeout, max_retries=args.max_retries, checkpoint_file=output_file, save_every=args.save_every
            )

    # Tong hop va xuat ket qua
    summary = compute_metrics_table(benchmark_data["results"])
    benchmark_data["summary"] = summary
    save_benchmark_results(benchmark_data, output_file)
    print(f"[SUCCESS] Hoan thanh toan bo quy trinh danh gia benchmark!")


if __name__ == "__main__":
    main()
