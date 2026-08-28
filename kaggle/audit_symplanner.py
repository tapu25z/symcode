"""Offline audit for legacy SymPlanner result files.

Usage:
    python kaggle/audit_symplanner.py result_symplanner_math500_n70_vastai.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "kaggle") not in sys.path:
    sys.path.insert(0, str(ROOT / "kaggle"))

from method.extractor import check_exact_match


def _rate(value: int, total: int) -> float:
    return round(100.0 * value / total, 2) if total else 0.0


def audit(path: str) -> dict[str, Any]:
    requested = Path(path)
    candidates = [requested]
    if "_symplanner_" in requested.name:
        candidates.append(requested.with_name(requested.name.replace("_symplanner_", "symplanner_")))
    resolved = next((candidate for candidate in candidates if candidate.exists()), requested)
    with resolved.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    results = list(data.get("results", {}).get("SymPlanner", []))
    total = len(results)
    normalized_correct = [check_exact_match(item.get("predicted"), item.get("ground_truth", "")) for item in results]
    false_passes = [item for item, correct in zip(results, normalized_correct) if item.get("verification_status") == "pass" and not correct]
    false_fails = [item for item, correct in zip(results, normalized_correct) if item.get("verification_status") == "fail" and correct]
    malformed = [item for item in results if item.get("planner_errors")]
    recovered = []
    for item in results:
        history = item.get("attempt_history") or []
        if len(history) > 1 and item.get("is_correct") and not check_exact_match(history[0].get("candidate_answer"), item.get("ground_truth", "")):
            recovered.append(item)
    token_values = [int(item.get("generated_tokens", 0)) for item in results]
    attempts = [int(item.get("attempts", 1)) for item in results]
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "config": data.get("config", {}),
        "total": total,
        "accuracy_percent": _rate(sum(bool(item.get("is_correct")) for item in results), total),
        "normalized_accuracy_percent": _rate(sum(normalized_correct), total),
        "verification_status": dict(Counter(item.get("verification_status", "unknown") for item in results)),
        "execution_status": dict(Counter(item.get("execution_status", "unknown") for item in results)),
        "false_pass_count": len(false_passes),
        "false_fail_count": len(false_fails),
        "planner_malformed_count": len(malformed),
        "repair_recovered_count": len(recovered),
        "avg_tokens": round(statistics.mean(token_values), 1) if token_values else 0.0,
        "median_tokens": statistics.median(token_values) if token_values else 0,
        "avg_attempts": round(statistics.mean(attempts), 2) if attempts else 0.0,
        "by_level": _group_metrics(results, "level_label"),
        "by_subject": _group_metrics(results, "subject"),
        "false_pass_examples": [
            {"index": results.index(item) + 1, "level": item.get("level"), "predicted": item.get("predicted"), "ground_truth": item.get("ground_truth")}
            for item in false_passes[:20]
        ],
    }


def _group_metrics(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        groups[str(item.get(key, "unknown"))].append(item)
    return {
        group: {
            "total": len(items),
            "correct": sum(bool(item.get("is_correct")) for item in items),
            "accuracy_percent": _rate(sum(bool(item.get("is_correct")) for item in items), len(items)),
            "normalized_correct": sum(check_exact_match(item.get("predicted"), item.get("ground_truth", "")) for item in items),
            "normalized_accuracy_percent": _rate(sum(check_exact_match(item.get("predicted"), item.get("ground_truth", "")) for item in items), len(items)),
            "avg_attempts": round(statistics.mean([int(item.get("attempts", 1)) for item in items]), 2),
        }
        for group, items in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a SymPlanner JSON result file without loading a model.")
    parser.add_argument("result_file")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()
    report = audit(args.result_file)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
