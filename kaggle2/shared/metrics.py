"""
Metrics computation and result persistence.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List


def compute_metrics_table(all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Tinh toan bang tong hop chi so danh gia toan dien: Overall, by Subject, by Difficulty, Subject x Difficulty.
    """
    summary_data = {}

    print("\n" + "=" * 90)
    print(f"{'TONG HOP KET QUA BENCHMARK':^90}")
    print("=" * 90)
    print(f"{'PHUONG PHAP':<14} | {'ACCURACY':<10} | {'DUNG/TONG':<15} | {'AVG TOKENS':<12} | {'AVG ATTEMPTS':<12} | {'EXEC RATE':<10} | {'VERIF PASS':<10}")
    print("-" * 90)

    for method, res_list in all_results.items():
        total = len(res_list)
        if total == 0:
            continue

        num_correct = sum(1 for r in res_list if r.get("is_correct"))
        acc = (num_correct / total) * 100.0
        avg_tokens = sum(r.get("generated_tokens", 0) for r in res_list) / total
        avg_attempts = sum(r.get("attempts", 1) for r in res_list) / total

        exec_applicable = [r for r in res_list if r.get("execution_status") not in ["not_applicable", None]]
        if exec_applicable:
            exec_success = sum(1 for r in exec_applicable if r.get("execution_status") == "success")
            exec_rate_str = f"{(exec_success / len(exec_applicable)) * 100.0:.1f}%"
        else:
            exec_rate_str = "N/A"

        verif_applicable = [r for r in res_list if r.get("verification_status") not in ["not_applicable", None]]
        if verif_applicable:
            verif_pass = sum(1 for r in verif_applicable if r.get("verification_status") == "pass")
            verif_rate_str = f"{(verif_pass / len(verif_applicable)) * 100.0:.1f}%"
        else:
            verif_rate_str = "N/A"

        print(f"{method:<14} | {acc:<9.2f}% | {f'{num_correct}/{total}':<15} | {avg_tokens:<12.1f} | {avg_attempts:<12.2f} | {exec_rate_str:<10} | {verif_rate_str:<10}")

        by_subject = {}
        subjects = sorted(list({r.get("subject", "unknown") for r in res_list}))
        for subj in subjects:
            subj_items = [r for r in res_list if r.get("subject", "unknown") == subj]
            s_corr = sum(1 for r in subj_items if r.get("is_correct"))
            s_tot = len(subj_items)
            by_subject[subj] = {
                "accuracy_percent": round((s_corr / s_tot) * 100.0, 2) if s_tot > 0 else 0.0,
                "correct": s_corr,
                "total": s_tot
            }

        by_difficulty = {}
        levels = sorted(list({r.get("level_label", "N/A") for r in res_list}))
        for lvl in levels:
            lvl_items = [r for r in res_list if r.get("level_label", "N/A") == lvl]
            l_corr = sum(1 for r in lvl_items if r.get("is_correct"))
            l_tot = len(lvl_items)
            by_difficulty[lvl] = {
                "accuracy_percent": round((l_corr / l_tot) * 100.0, 2) if l_tot > 0 else 0.0,
                "correct": l_corr,
                "total": l_tot
            }

        by_subject_x_difficulty = {}
        for subj in subjects:
            for lvl in levels:
                cell_items = [r for r in res_list if r.get("subject", "unknown") == subj and r.get("level_label", "N/A") == lvl]
                if cell_items:
                    c_corr = sum(1 for r in cell_items if r.get("is_correct"))
                    c_tot = len(cell_items)
                    key = f"{subj} | {lvl}"
                    by_subject_x_difficulty[key] = {
                        "subject": subj,
                        "difficulty": lvl,
                        "accuracy_percent": round((c_corr / c_tot) * 100.0, 2),
                        "correct": c_corr,
                        "total": c_tot
                    }

        summary_data[method] = {
            "accuracy_percent": round(acc, 2),
            "exact_match_count": num_correct,
            "total_samples": total,
            "avg_generated_tokens": round(avg_tokens, 1),
            "avg_attempts": round(avg_attempts, 2),
            "execution_success_rate": exec_rate_str,
            "verification_success_rate": verif_rate_str,
            "by_subject": by_subject,
            "by_difficulty": by_difficulty,
            "by_subject_x_difficulty": by_subject_x_difficulty
        }

    print("=" * 90 + "\n")
    return summary_data


def save_benchmark_results(results_data: Dict[str, Any], filepath: str) -> None:
    """Luu ket qua vao file JSON."""
    temp_file = filepath + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, filepath)
    print(f"[SUCCESS] Da luu ket qua benchmark tai: {filepath}")
