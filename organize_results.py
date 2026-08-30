from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

def load_dataset_metadata(root_dir: Path) -> Dict[str, Dict[str, Any]]:
    metadata = {}
    for ds in ["math500", "gsm8k"]:
        ds_file = root_dir / "kaggle2" / "data" / ds / "test.jsonl"
        if not ds_file.exists():
            ds_file = root_dir / "kaggle" / "data" / ds / "test.jsonl"
        if ds_file.exists():
            with open(ds_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        prob_key = item.get("problem", "").strip()
                        if prob_key:
                            metadata[prob_key] = {
                                "dataset": ds,
                                "level": item.get("level"),
                                "subject": item.get("subject", "Unknown"),
                                "answer": item.get("answer"),
                                "unique_id": item.get("unique_id")
                            }
    return metadata

def calculate_summary(samples: List[Dict[str, Any]], dataset_name: str, method_name: str, level_label: str) -> Dict[str, Any]:
    total = len(samples)
    if total == 0:
        return {
            "dataset": dataset_name,
            "method": method_name,
            "level": level_label,
            "total_samples": 0,
            "correct_count": 0,
            "accuracy_percent": 0.0,
            "by_subject": {}
        }
    correct_count = sum(1 for s in samples if s.get("is_correct"))
    acc = round((correct_count / total) * 100.0, 2)
    tokens_list = [s.get("generated_tokens") for s in samples if s.get("generated_tokens") is not None]
    avg_tokens = round(sum(tokens_list) / len(tokens_list), 1) if tokens_list else None
    attempts_list = [s.get("attempts") for s in samples if s.get("attempts") is not None]
    avg_attempts = round(sum(attempts_list) / len(attempts_list), 2) if attempts_list else None
    exec_applicable = [s for s in samples if s.get("execution_status") not in ["not_applicable", None]]
    if exec_applicable:
        exec_success = sum(1 for s in exec_applicable if s.get("execution_status") == "success")
        exec_rate_str = f"{(exec_success / len(exec_applicable)) * 100.0:.1f}%"
    else:
        exec_rate_str = "N/A"
    by_subject = {}
    subjects = sorted(list({s.get("subject", "Unknown") for s in samples}))
    for subj in subjects:
        subj_samples = [s for s in samples if s.get("subject", "Unknown") == subj]
        s_corr = sum(1 for s in subj_samples if s.get("is_correct"))
        s_tot = len(subj_samples)
        by_subject[subj] = {
            "correct": s_corr,
            "total": s_tot,
            "accuracy_percent": round((s_corr / s_tot) * 100.0, 2) if s_tot > 0 else 0.0
        }
    return {
        "dataset": dataset_name,
        "method": method_name,
        "level": level_label,
        "total_samples": total,
        "correct_count": correct_count,
        "accuracy_percent": acc,
        "avg_generated_tokens": avg_tokens,
        "avg_attempts": avg_attempts,
        "execution_success_rate": exec_rate_str,
        "by_subject": by_subject
    }

def export_hierarchy(base_output_dir: Path, organized_data: Dict[str, Dict[str, Dict[Any, List[Dict[str, Any]]]]]) -> None:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, methods in organized_data.items():
        dataset_dir = base_output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_summary = {"dataset": dataset_name, "methods": {}}
        for method_name, levels in methods.items():
            method_slug = method_name.lower().replace("+", "_plus").replace(" ", "_")
            method_dir = dataset_dir / method_slug
            method_dir.mkdir(parents=True, exist_ok=True)
            all_samples = []
            level_summaries = {}
            sorted_levels = sorted(levels.keys(), key=lambda x: (isinstance(x, str), str(x)))
            for lvl in sorted_levels:
                lvl_samples = levels[lvl]
                all_samples.extend(lvl_samples)
                lvl_folder_name = f"level_{lvl}" if str(lvl).isdigit() else f"level_{str(lvl).lower()}"
                lvl_dir = method_dir / lvl_folder_name
                lvl_dir.mkdir(parents=True, exist_ok=True)
                lvl_summary = calculate_summary(lvl_samples, dataset_name, method_name, f"Level {lvl}")
                level_summaries[f"Level {lvl}"] = lvl_summary
                with open(lvl_dir / "results.json", "w", encoding="utf-8") as f:
                    json.dump(lvl_samples, f, indent=2, ensure_ascii=False)
                with open(lvl_dir / "summary.json", "w", encoding="utf-8") as f:
                    json.dump(lvl_summary, f, indent=2, ensure_ascii=False)
            all_levels_dir = method_dir / "all_levels"
            all_levels_dir.mkdir(parents=True, exist_ok=True)
            all_summary = calculate_summary(all_samples, dataset_name, method_name, "All Levels")
            with open(all_levels_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(all_samples, f, indent=2, ensure_ascii=False)
            with open(all_levels_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(all_summary, f, indent=2, ensure_ascii=False)
            method_full_summary = {
                "dataset": dataset_name,
                "method": method_name,
                "overall": all_summary,
                "by_level": level_summaries
            }
            with open(method_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(method_full_summary, f, indent=2, ensure_ascii=False)
            dataset_summary["methods"][method_name] = method_full_summary
        with open(dataset_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(dataset_summary, f, indent=2, ensure_ascii=False)
        md_lines = [
            f"# Báo cáo Benchmark: {dataset_name.upper()}",
            "",
            "## Bảng tổng hợp theo Cấp độ (Level) và Phương pháp (Method)",
            "",
            "| Level | " + " | ".join(methods.keys()) + " |",
            "| :--- | " + " | ".join([":---:" for _ in methods]) + " |"
        ]
        all_unique_levels = sorted(
            list({lvl for m in methods.values() for lvl in m.keys()}),
            key=lambda x: (isinstance(x, str), str(x))
        )
        for lvl in all_unique_levels:
            row = [f"**Level {lvl}**"]
            for m_name in methods.keys():
                if lvl in methods[m_name]:
                    s = calculate_summary(methods[m_name][lvl], dataset_name, m_name, f"Level {lvl}")
                    row.append(f"{s['accuracy_percent']:.2f}% ({s['correct_count']}/{s['total_samples']})")
                else:
                    row.append("—")
            md_lines.append("| " + " | ".join(row) + " |")
        total_row = ["**TỔNG CỘNG**"]
        for m_name in methods.keys():
            m_all_samples = [s for lvl_samples in methods[m_name].values() for s in lvl_samples]
            s = calculate_summary(m_all_samples, dataset_name, m_name, "All Levels")
            total_row.append(f"**{s['accuracy_percent']:.2f}%** ({s['correct_count']}/{s['total_samples']})")
        md_lines.append("| " + " | ".join(total_row) + " |")
        with open(dataset_dir / "README.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
    print(f"\n[HOÀN TẤT] Toàn bộ kết quả đã được chia và lưu theo cấu trúc thư mục tại: {base_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Reorganize benchmark results into Dataset -> Method -> Level folder structure.")
    parser.add_argument("--input", default="result/all_results_combined.json", help="Path to combined results JSON.")
    parser.add_argument("--output-dir", default="results", help="Path to output base directory.")
    args = parser.parse_args()
    root_dir = Path(".").resolve()
    combined_json_path = root_dir / args.input
    output_dir = root_dir / args.output_dir
    if not combined_json_path.exists():
        print(f"[ERROR] Không tìm thấy file input: {combined_json_path}")
        return
    print(f"Đang đọc dữ liệu từ: {combined_json_path}")
    with open(combined_json_path, "r", encoding="utf-8") as f:
        combined_data = json.load(f)
    meta = load_dataset_metadata(root_dir)
    print(f"Đã nạp {len(meta)} mục metadata cho GSM8K & MATH-500.")
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    benchmarks = combined_data.get("benchmarks", {})
    if "gsm8k_lvl1_2_3" in benchmarks:
        gsm_results = benchmarks["gsm8k_lvl1_2_3"].get("results", {})
        for method, samples in gsm_results.items():
            for s in samples:
                prob = s.get("problem", "").strip()
                lvl = s.get("level")
                subj = s.get("subject")
                if prob in meta:
                    if lvl is None:
                        lvl = meta[prob].get("level")
                    if subj is None:
                        s["subject"] = meta[prob].get("subject", "Arithmetic")
                if lvl is None:
                    lvl = 1
                s["level"] = lvl
                organized["gsm8k"][method][lvl].append(s)
    if "math500_full_500" in benchmarks:
        m500_full_results = benchmarks["math500_full_500"].get("results", {})
        for method in ["Direct", "CoT"]:
            if method in m500_full_results:
                for s in m500_full_results[method]:
                    prob = s.get("problem", "").strip()
                    lvl = s.get("level")
                    subj = s.get("subject")
                    if prob in meta:
                        if lvl is None:
                            lvl = meta[prob].get("level")
                        if subj is None:
                            s["subject"] = meta[prob].get("subject", "Unknown")
                    if lvl is None:
                        lvl = "Unknown"
                    s["level"] = lvl
                    organized["math500"][method][lvl].append(s)
    if "math500_lvl1_3" in benchmarks:
        m13_results = benchmarks["math500_lvl1_3"].get("results", {})
        if "SymCode" in m13_results:
            for s in m13_results["SymCode"]:
                prob = s.get("problem", "").strip()
                lvl = s.get("level")
                subj = s.get("subject")
                if prob in meta:
                    if lvl is None:
                        lvl = meta[prob].get("level")
                    if subj is None:
                        s["subject"] = meta[prob].get("subject", "Unknown")
                if lvl is None:
                    lvl = "Unknown"
                s["level"] = lvl
                organized["math500"]["SymCode"][lvl].append(s)
    if "math500_lvl4" in benchmarks:
        m4_results = benchmarks["math500_lvl4"].get("results", {})
        sym_l4 = m4_results.get("SymCode+") or m4_results.get("SymCode")
        if sym_l4:
            for s in sym_l4:
                prob = s.get("problem", "").strip()
                lvl = 4
                subj = s.get("subject")
                if prob in meta and subj is None:
                    s["subject"] = meta[prob].get("subject", "Unknown")
                s["level"] = lvl
                organized["math500"]["SymCode"][lvl].append(s)
    if "math500_lvl5" in benchmarks:
        m5_results = benchmarks["math500_lvl5"].get("results", {})
        sym_l5 = m5_results.get("SymCode")
        if sym_l5:
            for s in sym_l5:
                prob = s.get("problem", "").strip()
                lvl = 5
                subj = s.get("subject")
                if prob in meta and subj is None:
                    s["subject"] = meta[prob].get("subject", "Unknown")
                s["level"] = lvl
                organized["math500"]["SymCode"][lvl].append(s)
    export_hierarchy(output_dir, organized)

if __name__ == "__main__":
    main()
