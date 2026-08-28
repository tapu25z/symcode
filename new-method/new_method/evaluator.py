"""Benchmark evaluator compatible with the repository's existing result schema."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

from .adapters import Legacy7BCoderAdapter, LegacySandboxAdapter, StageTokenBudgets
from .config import ABLATIONS
from .pipeline import SymPlannerIRPipeline
from .scoring import check_math500_equivalence


def _legacy_scoring_helpers():
    kaggle_dir = Path(__file__).resolve().parents[2] / "kaggle"
    if str(kaggle_dir) not in sys.path:
        sys.path.insert(0, str(kaggle_dir))
    from method import check_exact_match, extract_ground_truth, normalize_answer_str
    return extract_ground_truth, check_exact_match, normalize_answer_str


def _load_checkpoint(path: str | None, method_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path or not os.path.exists(path):
        return {}, []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data, list(data.get("results", {}).get(method_name, []))
    except Exception:
        return {}, []


def _save_checkpoint(path: str | None, method_name: str, results: list[dict[str, Any]], metadata: Mapping[str, Any] | None = None) -> None:
    if not path:
        return
    data, _ = _load_checkpoint(path, method_name)
    data.setdefault("results", {})[method_name] = results
    if metadata:
        data.setdefault("config", {}).update(dict(metadata))
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def evaluate_ir_variant(
    dataset: list[dict[str, Any]],
    llm_runner: Any,
    sandbox_executor: Callable[..., Mapping[str, Any]],
    variant: str = "IR",
    timeout: int = 15,
    max_retries: int = 2,
    checkpoint_file: str | None = None,
    save_every: int = 5,
    token_budgets: StageTokenBudgets | None = None,
    ground_truth_fn: Callable[[Any], str] | None = None,
    match_fn: Callable[[Any, Any], bool] | None = None,
) -> list[dict[str, Any]]:
    if variant not in ABLATIONS:
        raise ValueError(f"unknown variant {variant!r}")
    legacy_match = legacy_normalize = None
    if ground_truth_fn is None or match_fn is None:
        default_gt, legacy_match, legacy_normalize = _legacy_scoring_helpers()
        ground_truth_fn = ground_truth_fn or default_gt
    _, results = _load_checkpoint(checkpoint_file, variant)
    completed = {item.get("problem") for item in results}
    model = Legacy7BCoderAdapter(llm_runner, token_budgets)
    sandbox = LegacySandboxAdapter(sandbox_executor, timeout=timeout)
    pipeline = SymPlannerIRPipeline(model, sandbox, max_repairs=max_retries, ablation=variant)
    new_count = 0

    print(f"\n==================== Bat dau danh gia Ablation: {variant} ({len(dataset)} mau) ====================")
    if completed:
        print(f"[INFO] Da hoan thanh truoc do: {len(completed)}/{len(dataset)} mau.")
    sys.stdout.flush()

    for idx, item in enumerate(dataset):
        question = str(item.get("question") or item.get("problem") or "")
        if question in completed:
            continue
        
        print(f"\n[INFO] [{idx+1}/{len(dataset)}] [Lvl {item.get('level')} {item.get('subject')}] Running {variant}: {question[:80]}...")
        sys.stdout.flush()
        
        snapshot = model.snapshot()
        started = time.perf_counter()
        trace = pipeline.run(question)
        latency = time.perf_counter() - started
        calls = model.calls_since(snapshot)
        final = trace.get("final") or {}
        execution = final.get("execution") if isinstance(final, Mapping) else {}
        execution = execution if isinstance(execution, Mapping) else {}
        verification = final.get("verification") if isinstance(final, Mapping) else {}
        verification = verification if isinstance(verification, Mapping) else {}
        predicted = execution.get("answer")
        raw_gold = item.get("raw") or item.get("answer") or ""
        ground_truth = ground_truth_fn(raw_gold)
        stage_tokens: dict[str, int] = defaultdict(int)
        for call in calls:
            stage_tokens[str(call["stage"])] += int(call["generated_tokens"])
        attempts = list(trace.get("attempts") or [])
        first_status = attempts[0].get("verification", {}).get("status") if attempts else trace.get("status")
        feedback = verification.get("feedback") or trace.get("schema_errors") or []
        if match_fn is not None:
            is_correct = bool(match_fn(predicted, ground_truth))
        else:
            is_correct = check_math500_equivalence(predicted, execution.get("canonical_answer"), ground_truth, legacy_match, legacy_normalize)
        result = {
            "problem": question,
            "subject": item.get("subject", "unknown"),
            "level": item.get("level"),
            "level_label": item.get("level_label", "N/A"),
            "ground_truth": ground_truth,
            "predicted": predicted,
            "is_correct": is_correct,
            "generated_tokens": sum(int(call["generated_tokens"]) for call in calls),
            "stage_tokens": dict(stage_tokens),
            "latency_seconds": round(latency, 4),
            "attempts": len(attempts),
            "execution_status": execution.get("_sandbox_status", "invalid_ir" if trace.get("status") == "invalid_ir" else "unknown"),
            "verification_status": trace.get("status"),
            "verification_feedback": feedback,
            "stdout": execution.get("_stdout", ""),
            "traceback": execution.get("_traceback"),
            "extracted_code": trace.get("code"),
            "planner_note": json.dumps(trace.get("ir"), ensure_ascii=False, default=str),
            "raw_output": calls[-1]["response"] if calls else "",
            "raw_outputs": [call["response"] for call in calls],
            "attempt_history": attempts,
            "pipeline_trace": trace,
            "variant": variant,
            "answer_type": execution.get("answer_type"),
            "invalid_ir": trace.get("status") == "invalid_ir",
            "repair_recovered": first_status != "pass" and trace.get("status") == "pass",
        }
        results.append(result)
        completed.add(question)
        new_count += 1

        match_icon = "OK" if is_correct else "FAIL"
        print(f"  -> {match_icon} | KQ: `{predicted}` | GT: `{ground_truth}` | Status: {trace.get('status')} | {latency:.1f}s ({result['generated_tokens']} tokens)")
        sys.stdout.flush()

        if save_every and new_count % save_every == 0:
            _save_checkpoint(checkpoint_file, variant, results, {"last_saved": time.strftime("%Y-%m-%d %H:%M:%S")})
            print(f"[INFO] Da luu checkpoint {variant}: {len(results)}/{len(dataset)} mau.")
            sys.stdout.flush()
            
    _save_checkpoint(checkpoint_file, variant, results, {"ir_variants": sorted(ABLATIONS)})
    return results


def compute_ir_diagnostics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if not total:
        return {"total": 0}
    return {
        "total": total,
        "accuracy": round(100 * sum(bool(item.get("is_correct")) for item in results) / total, 2),
        "invalid_ir_rate": round(100 * sum(bool(item.get("invalid_ir")) for item in results) / total, 2),
        "repair_recovery_rate": round(100 * sum(bool(item.get("repair_recovered")) for item in results) / total, 2),
        "verification_pass_rate": round(100 * sum(item.get("verification_status") == "pass" for item in results) / total, 2),
        "avg_generated_tokens": round(sum(int(item.get("generated_tokens", 0)) for item in results) / total, 1),
        "avg_latency_seconds": round(sum(float(item.get("latency_seconds", 0.0)) for item in results) / total, 3),
    }
