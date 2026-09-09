"""
Script reset ket qua SymPlanner trong file checkpoint JSON de cac phuong phap khac (Direct, CoT, SymCode) duoc SKIP,
con SymPlanner se chay lai tu dau voi pipeline moi da nang cap.

Usage:
    python reset_symplanner.py
    python reset_symplanner.py results/results_full_math500_qwen3_4b.json
"""
import sys
import json
import os
import shutil


def reset_symplanner(filepath: str = "results/results_full_math500_qwen3_4b.json"):
    if not os.path.exists(filepath):
        print(f"[ERROR] Khong tim thay file: {filepath}")
        return

    # Backup file truoc khi sua
    backup_path = filepath + ".bak"
    shutil.copy2(filepath, backup_path)
    print(f"[INFO] Da tao file backup tai: {backup_path}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    old_sp_count = len(results.get("SymPlanner", []))

    # Reset SymPlanner
    results["SymPlanner"] = []

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Da reset SymPlanner ({old_sp_count} -> 0 mau).")
    print(f"[INFO] Trang thai cac phuong phap trong file {filepath}:")
    for method, items in results.items():
        status = "SE CHAY LAI (0 mau)" if method == "SymPlanner" else f"SE DUOC SKIP ({len(items)} mau da hoan thanh)"
        print(f"   * {method:<12}: {status}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "results/results_full_math500_qwen3_4b.json"
    reset_symplanner(target)
