"""
run_notebook.py - Module dieu phoi thuc thi Benchmark cho Notebook Kaggle/Jupyter.
Quan ly 4 phuong phap rieng biet trong thu muc methods/:
- Baselines: Direct (methods/direct/), CoT (methods/cot/), SymCode (methods/symcode/)
- Ours: SymPlan (methods/symplan/)
"""

from __future__ import annotations

import os
import gc
import sys
import glob
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Quan ly phan manh bo nho CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def locate_directory(target_name: str) -> str:
    """Tim kiem thu muc muc tieu trong moi truong Kaggle hoac workspace cuc bo."""
    for search_root in ["/kaggle/input", ".", "..", "kaggle2"]:
        if os.path.exists(search_root):
            for dirpath, dirnames, _ in os.walk(search_root):
                if target_name in dirnames:
                    return os.path.abspath(dirpath)
    return os.path.abspath(".")


def setup_environment(quiet: bool = False) -> str:
    """Cau hinh sys.path va cai dat cac thu vien can thiet."""
    root_dir = locate_directory("methods")
    if not root_dir or root_dir == os.path.abspath("."):
        root_dir = locate_directory("shared")

    for p in [root_dir, os.path.join(root_dir, "shared"), os.path.join(root_dir, "methods")]:
        if p and os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)

    if not quiet:
        print(f"[INFO] Thu muc goc du an: {root_dir}")

    req_matches = glob.glob("/kaggle/input/**/requirements.txt", recursive=True) or ["requirements.txt"]
    req_file = req_matches[0] if (req_matches and os.path.exists(req_matches[0])) else os.path.join(root_dir, "requirements.txt")
    if os.path.exists(req_file):
        os.system(f'pip install -q -r "{req_file}"')
    else:
        os.system("pip install -q bitsandbytes accelerate transformers sympy datasets tqdm pandas")

    try:
        import torch
        if not quiet:
            print(f"[INFO] PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB VRAM)")
    except ImportError:
        pass

    return root_dir


def find_data_file(subpath_pattern: str, root_dir: str = ".") -> str:
    """Dinh vi file du lieu trong Kaggle Input hoac thu muc cuc bo."""
    matches = glob.glob(f"/kaggle/input/**/{subpath_pattern}", recursive=True)
    if matches:
        return os.path.abspath(matches[0])
    local_p = os.path.join(root_dir, subpath_pattern)
    if os.path.exists(local_p):
        return os.path.abspath(local_p)
    if os.path.exists(subpath_pattern):
        return os.path.abspath(subpath_pattern)
    return subpath_pattern


def build_config(
    dataset_name: str = "math500",
    dataset_path: Optional[str] = None,
    methods: Optional[List[str]] = None,
    num_samples: Optional[int] = 5,
    filter_levels: Optional[List[int]] = None,
    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    load_in_4bit: bool = True,
    max_input_tokens: int = 6144,
    max_new_tokens: int = 1800,
    temperature: float = 0.0,
    timeout: int = 15,
    max_retries: int = 2,
    max_ir_retries: int = 1
) -> Dict[str, Any]:
    """Xay dung bang cau hinh benchmark chuan hoa."""
    root_dir = setup_environment(quiet=True)

    if methods is None:
        methods = ["Direct", "CoT", "SymCode", "SymPlan"]

    if not dataset_path:
        subpath = "data/math500/test.jsonl" if dataset_name == "math500" else "data/gsm8k/test.jsonl"
        dataset_path = find_data_file(subpath, root_dir)

    lvl_str = ("_lvl" + "_".join(str(lvl) for lvl in sorted(filter_levels))) if filter_levels else ""
    out_name = f"{dataset_name}{lvl_str}_benchmark_results.json"
    output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    output_file = os.path.join(output_dir, out_name)

    if num_samples is not None and num_samples <= 10 and os.path.exists(output_file):
        try:
            os.remove(output_file)
            print(f"[INFO] Da lam moi file ket qua test nhanh: {output_file}")
        except Exception:
            pass

    from methods.symplan.adapters import StageTokenBudgets
    token_budgets = StageTokenBudgets(
        extractor=1400,
        ir_repair=1400,
        codegen=1800,
        code_repair=1800
    )

    config = {
        "model_id": model_id,
        "load_in_4bit": load_in_4bit,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": 42,
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "filter_levels": filter_levels,
        "num_samples": num_samples,
        "methods_to_run": methods,
        "code_exec_timeout": timeout,
        "max_retries": max_retries,
        "max_ir_retries": max_ir_retries,
        "output_file": output_file,
        "stage_token_budgets": token_budgets.__dict__,
        "save_every": 1 if (num_samples and num_samples <= 10) else 5
    }

    print("=" * 75)
    print(f"[INFO] Mo hinh: {config['model_id']}")
    print(f"[INFO] Tap du lieu: {config['dataset_name'].upper()} ({config['dataset_path']})")
    print(f"[INFO] Bo loc do kho: {config['filter_levels']}")
    print(f"[INFO] So luong mau: {config['num_samples'] if config['num_samples'] is not None else 'TOAN BO'}")
    print(f"[INFO] Danh sach phuong phap: {config['methods_to_run']}")
    print(f"[INFO] File luu ket qua: {config['output_file']}")
    print("=" * 75)

    return config


def run_benchmark_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Thuc thi toan bo quy trinh benchmark qua cac package rieng biet trong methods/."""
    import torch
    import pandas as pd
    from IPython.display import display

    from shared import load_dataset_file, compute_metrics_table, save_benchmark_results, execute_code_safely
    from methods.direct import evaluate_direct
    from methods.cot import evaluate_cot
    from methods.symcode import evaluate_symcode
    from methods.symplan import build_legacy_7b_runner, evaluate_symplan, evaluate_ir_variant, compute_ir_diagnostics, ABLATIONS
    from methods.symplan.adapters import StageTokenBudgets

    # 1. Don dep bo nho VRAM
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Nap Dataset
    dataset = load_dataset_file(
        config["dataset_path"],
        split="test",
        num_samples=config["num_samples"],
        filter_levels=config.get("filter_levels")
    )
    if not dataset:
        raise ValueError(f"Khong the nap du lieu tu: {config['dataset_path']}")

    # 3. Khoi tao Mo hinh
    runner = build_legacy_7b_runner(
        model_id=config["model_id"],
        load_in_4bit=config["load_in_4bit"],
        max_new_tokens=config["max_new_tokens"],
        max_input_tokens=config["max_input_tokens"],
        temperature=config["temperature"]
    )

    token_budgets = StageTokenBudgets(**config.get("stage_token_budgets", {}))

    # 4. Quan ly Checkpoint & Auto-Resume
    out_f = config["output_file"]
    save_n = config.get("save_every", 5)

    if os.path.exists(out_f):
        try:
            with open(out_f, "r", encoding="utf-8") as f:
                benchmark_data = json.load(f)
        except Exception:
            benchmark_data = {"config": config, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": {}, "summary": {}}
    else:
        benchmark_data = {"config": config, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": {}, "summary": {}}

    benchmark_data["config"] = config
    benchmark_data.setdefault("results", {})

    # 5. Thuc thi tung phuong phap rieng biet
    for method in config["methods_to_run"]:
        print("\n" + "=" * 75)
        print(f"[RUNNING METHOD] ===> {method}")
        print("=" * 75)

        if method == "Direct":
            benchmark_data["results"]["Direct"] = evaluate_direct(
                dataset, runner, checkpoint_file=out_f, save_every=save_n
            )
        elif method == "CoT":
            benchmark_data["results"]["CoT"] = evaluate_cot(
                dataset, runner, checkpoint_file=out_f, save_every=save_n
            )
        elif method == "SymCode":
            benchmark_data["results"]["SymCode"] = evaluate_symcode(
                dataset, runner, timeout=config["code_exec_timeout"],
                max_retries=config["max_retries"], checkpoint_file=out_f, save_every=save_n
            )
        elif method in ["SymPlan", "IR-Full"]:
            benchmark_data["results"][method] = evaluate_symplan(
                dataset, runner, execute_code_safely, variant="SymPlan",
                timeout=config["code_exec_timeout"], max_retries=config["max_retries"],
                max_ir_retries=config["max_ir_retries"], checkpoint_file=out_f,
                save_every=save_n, token_budgets=token_budgets
            )
        elif method in ABLATIONS:
            benchmark_data["results"][method] = evaluate_ir_variant(
                dataset, runner, execute_code_safely, variant=method,
                timeout=config["code_exec_timeout"], max_retries=config["max_retries"],
                max_ir_retries=config["max_ir_retries"], checkpoint_file=out_f,
                save_every=save_n, token_budgets=token_budgets
            )
        else:
            print(f"[WARN] Khong nhan dien phuong phap: {method}. Bo qua...")

    # 6. Tong hop chi so
    summary = compute_metrics_table(benchmark_data["results"])
    benchmark_data["summary"] = summary
    benchmark_data["ir_diagnostics"] = {
        m: compute_ir_diagnostics(benchmark_data["results"].get(m, []))
        for m in list(ABLATIONS.keys()) + ["SymPlan", "IR-Full"]
        if m in benchmark_data["results"]
    }
    save_benchmark_results(benchmark_data, config["output_file"])

    # 7. Hien thi bang ket qua da chieu
    print("\n" + "=" * 75)
    print("--- BANG 1: TONG HOP CHI SO TOAN DIEN (OVERALL METRICS) ---")
    print("=" * 75)
    df_overall = pd.DataFrame.from_dict({
        m: {
            "Accuracy (%)": v["accuracy_percent"],
            "Exact Match": f"{v['exact_match_count']}/{v['total_samples']}",
            "Avg Tokens": v["avg_generated_tokens"],
            "Avg Attempts": v["avg_attempts"],
            "Exec Success": v["execution_success_rate"],
            "Verif Pass": v["verification_success_rate"]
        }
        for m, v in summary.items()
    }, orient="index")
    display(df_overall)

    subj_diff_rows = []
    for m, v in summary.items():
        for key, cell in v.get("by_subject_x_difficulty", {}).items():
            subj_diff_rows.append({
                "Method": m,
                "Subject": cell["subject"],
                "Difficulty": cell["difficulty"],
                "Correct": cell["correct"],
                "Total": cell["total"],
                "Accuracy (%)": cell["accuracy_percent"]
            })

    if subj_diff_rows:
        print("\n" + "=" * 75)
        print("--- BANG 2: PHAN RA THEO SUBJECT x DIFFICULTY ---")
        print("=" * 75)
        df_subj_diff = pd.DataFrame(subj_diff_rows)
        display(df_subj_diff)

    if benchmark_data.get("ir_diagnostics"):
        print("\n" + "=" * 75)
        print("--- BANG 3: CHAN DOAN IR VA DO TIN CAY (IR DIAGNOSTICS) ---")
        print("=" * 75)
        df_diag = pd.DataFrame.from_dict({
            m: {
                "Accuracy (%)": v.get("accuracy", 0.0),
                "Invalid IR (%)": v.get("invalid_ir_rate", 0.0),
                "Repair Recovered (%)": v.get("repair_recovery_rate", 0.0),
                "Verif Pass (%)": v.get("verification_pass_rate", 0.0),
                "Avg Latency (s)": v.get("avg_latency_seconds", 0.0),
                "Avg Tokens": v.get("avg_generated_tokens", 0.0)
            }
            for m, v in benchmark_data["ir_diagnostics"].items()
        }, orient="index")
        display(df_diag)

    print(f"\n[SUCCESS] Hoan thanh danh gia! File ket qua: {config['output_file']}")
    return benchmark_data
