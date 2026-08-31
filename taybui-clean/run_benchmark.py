"""
Script thuc thi Benchmark danh gia nang luc suy luan toan hoc (Direct, CoT, SymCode).
Ho tro chay truc tiep qua Command Line Interface (CLI) voi day du cac tuy chon tham so.
"""

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Optional

from method import (
    LLMRunner,
    load_dataset_file,
    compute_metrics_table,
    save_benchmark_results
)
from method.direct import evaluate as evaluate_direct
from method.cot import evaluate as evaluate_cot
from method.symcode import evaluate as evaluate_symcode
from method.symplanner import evaluate as evaluate_symplanner


SYMPLANNER_ABLATION_METHODS = {
    "SymPlanner": "full",
    "SymPlannerExtractOnly": "extract_only",
    "SymPlannerPlanOnly": "plan_only",
    "SymPlannerNoModules": "none",
}


MODEL_PRESETS = {
    "qwen2.5-coder-7b": {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "default_enable_thinking": None,
    },
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "default_enable_thinking": False,
    },
    "llama3-8b": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "default_enable_thinking": None,
    },
}


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
        choices=[
            "Direct",
            "CoT",
            "SymCode",
            "SymPlanner",
            "SymPlannerExtractOnly",
            "SymPlannerPlanOnly",
            "SymPlannerNoModules",
        ],
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
        "--model-preset",
        type=str,
        default=None,
        choices=sorted(MODEL_PRESETS.keys()),
        help="Preset model tien loi. Neu dung, preset se ghi de --model-id."
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
        "--max-input-tokens",
        type=int,
        default=2560,
        help="So token toi da cua prompt dau vao sau khi ap chat template."
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
        default=15,
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
    parser.add_argument(
        "--run-order",
        type=str,
        default="by-problem",
        choices=["by-problem", "by-method"],
        help="Thu tu chay: 'by-problem' = moi cau chay lan luot tung phuong phap; 'by-method' = chay het dataset cho tung phuong phap."
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="default_enable_thinking",
        action="store_true",
        default=None,
        help="Bat thinking trong chat template neu model/tokenizer ho tro."
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="default_enable_thinking",
        action="store_false",
        help="Tat thinking trong chat template neu model/tokenizer ho tro."
    )
    return parser.parse_args()


def _write_json_atomic(data: Dict[str, Any], filepath: str) -> None:
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    temp_file = filepath + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, filepath)


def _load_benchmark_data(output_file: str, config: Dict[str, Any]) -> Dict[str, Any]:
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                benchmark_data = json.load(f)
        except Exception:
            benchmark_data = {}
    else:
        benchmark_data = {}

    benchmark_data.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    benchmark_data["config"] = config
    benchmark_data.setdefault("results", {})
    benchmark_data.setdefault("summary", {})
    benchmark_data.setdefault("live_accuracy", {})
    benchmark_data.setdefault("live_accuracy_by_problem", [])
    return benchmark_data


def _accuracy_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    accuracy = (correct / total) * 100.0 if total else 0.0
    return {
        "accuracy_percent": round(accuracy, 2),
        "correct": correct,
        "total": total
    }


def _refresh_live_accuracy(benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
    live_accuracy = {
        method: _accuracy_stats(results)
        for method, results in benchmark_data.get("results", {}).items()
    }
    benchmark_data["live_accuracy"] = live_accuracy
    return live_accuracy


def _print_live_accuracy(live_accuracy: Dict[str, Dict[str, Any]]) -> None:
    if not live_accuracy:
        return
    parts = []
    for method, stats in live_accuracy.items():
        parts.append(
            f"{method}: {stats['accuracy_percent']:.2f}% ({stats['correct']}/{stats['total']})"
        )
    print("[LIVE ACC] " + " | ".join(parts), flush=True)


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _print_progress(
    completed: int,
    total: int,
    start_time: float,
) -> None:
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - completed)
    eta = remaining / rate if rate > 0 else None
    percent = (completed / total) * 100.0 if total else 100.0
    print(
        f"[PROGRESS] {completed}/{total} ({percent:.1f}%) | "
        f"{rate:.3f} it/s | elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)}",
        flush=True,
    )


def _find_result_for_problem(results: List[Dict[str, Any]], question: str) -> Optional[Dict[str, Any]]:
    for record in results:
        if record.get("problem") == question:
            return record
    return None


def _has_result_for_problem(results: List[Dict[str, Any]], question: str) -> bool:
    return _find_result_for_problem(results, question) is not None


def _record_problem_accuracy_snapshot(
    benchmark_data: Dict[str, Any],
    problem_index: int,
    total_problems: int,
    question: str,
    live_accuracy: Dict[str, Dict[str, Any]],
) -> None:
    snapshots = benchmark_data.setdefault("live_accuracy_by_problem", [])
    snapshots = [s for s in snapshots if s.get("problem") != question]
    snapshots.append({
        "problem_index": problem_index,
        "total_problems": total_problems,
        "problem": question,
        "live_accuracy": live_accuracy,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    benchmark_data["live_accuracy_by_problem"] = snapshots


def _print_problem_accuracy_snapshot(
    problem_index: int,
    total_problems: int,
    live_accuracy: Dict[str, Dict[str, Any]],
) -> None:
    if not live_accuracy:
        return
    parts = []
    for method, stats in live_accuracy.items():
        parts.append(
            f"{method}: {stats['accuracy_percent']:.2f}% ({stats['correct']}/{stats['total']})"
        )
    print(
        f"[LIVE ACC PROBLEM {problem_index}/{total_problems}] " + " | ".join(parts),
        flush=True
    )


def _run_single_method(
    method: str,
    item: Dict[str, Any],
    llm: LLMRunner,
    args: argparse.Namespace,
    output_file: str,
) -> List[Dict[str, Any]]:
    if method == "Direct":
        return evaluate_direct(
            [item], llm, checkpoint_file=output_file, save_every=1, verbose=False
        )
    if method == "CoT":
        return evaluate_cot(
            [item], llm, checkpoint_file=output_file, save_every=1, verbose=False
        )
    if method == "SymCode":
        return evaluate_symcode(
            [item], llm, timeout=args.timeout, max_retries=args.max_retries,
            checkpoint_file=output_file, save_every=1, verbose=False
        )
    if method in SYMPLANNER_ABLATION_METHODS:
        return evaluate_symplanner(
            [item], llm, timeout=args.timeout, max_retries=args.max_retries,
            checkpoint_file=output_file, save_every=1, verbose=False,
            ablation=SYMPLANNER_ABLATION_METHODS[method], method_name=method
        )
    raise ValueError(f"Phuong phap khong hop le: {method}")


def _run_by_problem(
    dataset: List[Dict[str, Any]],
    llm: LLMRunner,
    args: argparse.Namespace,
    config: Dict[str, Any],
    output_file: str,
) -> Dict[str, Any]:
    benchmark_data = _load_benchmark_data(output_file, config)
    _write_json_atomic(benchmark_data, output_file)

    total = len(dataset)
    start_time = time.time()
    print("\n==================== Bat dau chay theo tung cau ====================")
    for index, item in enumerate(dataset, start=1):
        question = item["question"]
        short_question = " ".join(str(question).split())[:120]
        print(f"\n[PROBLEM {index}/{total}] {short_question}", flush=True)

        for method in args.methods:
            method_results = benchmark_data.setdefault("results", {}).setdefault(method, [])
            if _has_result_for_problem(method_results, question):
                print(f"[SKIP] {method}: da co ket qua trong checkpoint.", flush=True)
            else:
                print(f"[RUN] {method}...", flush=True)
                method_results = _run_single_method(method, item, llm, args, output_file)
                benchmark_data["results"][method] = method_results

            current_result = _find_result_for_problem(benchmark_data["results"].get(method, []), question)
            if current_result:
                verdict = "DUNG" if current_result.get("is_correct") else "SAI"
                print(f"[{method}] cau nay: {verdict} | pred={current_result.get('predicted')} | gt={current_result.get('ground_truth')}", flush=True)

        live_accuracy = _refresh_live_accuracy(benchmark_data)
        _record_problem_accuracy_snapshot(benchmark_data, index, total, question, live_accuracy)
        _write_json_atomic(benchmark_data, output_file)
        _print_problem_accuracy_snapshot(index, total, live_accuracy)
        _print_progress(index, total, start_time)

    return benchmark_data


def main():
    args = parse_args()

    preset = MODEL_PRESETS.get(args.model_preset) if args.model_preset else None
    if preset:
        args.model_id = preset["model_id"]
        if args.default_enable_thinking is None:
            args.default_enable_thinking = preset["default_enable_thinking"]

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
        output_file = os.path.join("results", f"{args.dataset}{lvl_str}{tail_str}_results.json")

    config = {
        "model_id": args.model_id,
        "load_in_4bit": args.load_in_4bit,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "temperature": args.temperature,
        "default_enable_thinking": args.default_enable_thinking,
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
        "save_every": args.save_every,
        "run_order": args.run_order
    }

    print("=" * 75)
    print(f"[INFO] Mo hinh: {config['model_id']}")
    print(f"[INFO] Tap du lieu: {config['dataset_name'].upper()} ({config['dataset_path']})")
    print(f"[INFO] Loc do kho: {config['filter_levels']}")
    print(f"[INFO] Mau moi level: {config['per_level_samples']}")
    print(f"[INFO] Lat nguoc lay tu cuoi (tail): {config['tail']}")
    print(f"[INFO] So mau danh gia: {config['num_samples'] if config['num_samples'] is not None else 'TOAN BO'}")
    print(f"[INFO] Cac phuong phap: {config['methods_to_run']}")
    print(f"[INFO] Thu tu chay: {config['run_order']}")
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
    if LLMRunner is None:
        print("[ERROR] Khong import duoc LLMRunner. Hay cai dat dependencies: pip install -r requirements.txt")
        sys.exit(1)

    llm = LLMRunner(
        model_id=args.model_id,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_input_tokens=args.max_input_tokens,
        default_enable_thinking=args.default_enable_thinking
    )

    if args.run_order == "by-problem":
        benchmark_data = _run_by_problem(dataset, llm, args, config, output_file)
    else:
        benchmark_data = _load_benchmark_data(output_file, config)
        # Thuc thi tung phuong phap
        for method in args.methods:
            if method == "Direct":
                benchmark_data["results"]["Direct"] = evaluate_direct(
                    dataset, llm, checkpoint_file=output_file, save_every=args.save_every
                )
            elif method == "CoT":
                benchmark_data["results"]["CoT"] = evaluate_cot(
                    dataset, llm, checkpoint_file=output_file, save_every=args.save_every
                )
            elif method == "SymCode":
                benchmark_data["results"]["SymCode"] = evaluate_symcode(
                    dataset, llm, timeout=args.timeout, max_retries=args.max_retries, checkpoint_file=output_file, save_every=args.save_every
                )
            elif method in SYMPLANNER_ABLATION_METHODS:
                benchmark_data["results"][method] = evaluate_symplanner(
                    dataset, llm, timeout=args.timeout, max_retries=args.max_retries,
                    checkpoint_file=output_file, save_every=args.save_every,
                    ablation=SYMPLANNER_ABLATION_METHODS[method], method_name=method
                )
            live_accuracy = _refresh_live_accuracy(benchmark_data)
            _write_json_atomic(benchmark_data, output_file)
            _print_live_accuracy(live_accuracy)

    # Tong hop va xuat ket qua
    summary = compute_metrics_table(benchmark_data["results"])
    benchmark_data["summary"] = summary
    _refresh_live_accuracy(benchmark_data)
    save_benchmark_results(benchmark_data, output_file)
    print(f"[SUCCESS] Hoan thanh toan bo quy trinh danh gia benchmark!")


if __name__ == "__main__":
    main()
