#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runner orchestration cho pipeline inference.

File này nối các module lại với nhau theo thứ tự: few-shot -> Module 2 ->
Module 3 -> Module 4/execute/debug -> verify -> ghi output/report.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .fewshot import select_few_shot_for_module2, select_few_shot_for_module4
from .io import append_jsonl, read_jsonl_objects, reset_file, write_report
from .module2 import build_module2_few_shot_example, detect_answer_type, generate_module2, generate_problem_type, infer_problem_type
from .module3_units import process_module3
from .module4_codegen import build_module4_few_shot_example, build_module4_payload, generate_and_execute_with_debug
from .rag import vote_problem_type_from_rag
from .verify import verify_against_gold
from .debug_trace import compact_examples, get_events, log_json, reset_events


def classify_pipeline_result(
    execution: Dict[str, Any],
    verification: Dict[str, Any],
    answer_type: str,
) -> Tuple[str, str, str]:
    """
    Convert raw execution/verification into review-friendly labels.

    Returns:
      pipeline_status, failure_stage, debug_label
    """
    if execution.get("status") != "pass":
        err = execution.get("error_type") or "execution_error"
        if err == "SafetyError":
            label = "unsafe_code"
        elif err == "OutputFormatError":
            label = "no_boxed_output"
        elif err == "TimeoutExpired":
            label = "timeout"
        else:
            label = "runtime_error"
        return (
            "fail",
            "code_execution",
            label,
        )

    vstatus = verification.get("status")
    if vstatus == "pass":
        return (
            "pass",
            "none",
            "pass",
        )

    if vstatus == "skip":
        reason = str(verification.get("reason", ""))
        label = "manual_review" if "manual_review" in reason else "missing_or_unparsed_gold"
        return (
            "skip",
            "verification",
            label,
        )

    # vstatus == fail or unknown verification issue
    label = "answer_mismatch"
    reason = str(verification.get("reason", ""))
    if "unit" in reason.lower():
        label = "unit_mismatch"
    elif answer_type == "numeric_compute":
        label = "numeric_answer_mismatch"

    return (
        "fail",
        "verification",
        label,
    )


def build_output_obj(
    record: Dict[str, Any],
    answer_type: str,
    mod2_json: Dict[str, Any],
    mod3_json: Dict[str, Any],
    mod4_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build one review/evaluation record.

    Main inference purpose:
      - measure how good the pipeline is
      - locate failure stage
      - keep enough artifacts for manual review
    """
    verification = verify_against_gold(record, mod4_result["execution"], answer_type)

    pipeline_status, failure_stage, debug_label = (
        classify_pipeline_result(mod4_result["execution"], verification, answer_type)
    )

    return {
        "id":                  record.get("id", ""),
        "detected_problem_type": record.get("_detected_problem_type",
                                             infer_problem_type(record)),
        "target_answer_type":  answer_type,
        "pipeline_status":     pipeline_status,
        "failure_stage":       failure_stage,
        "debug_label":         debug_label,
        "question":            record.get("question", ""),
        "original_answer":     record.get("answer", ""),
        "original_unit":       record.get("unit", ""),
        "module2_extract":     mod2_json,
        "module3_si_extract":  mod3_json,
        "module4_symcode":     mod4_result["code"],
        "execution":           mod4_result["execution"],
        "verification":        verification,
        "debugged":            mod4_result["debugged"],
        "given_audit":         mod4_result.get("given_audit", {}),
        "given_audit_history": mod4_result.get("given_audit_history", []),
        "debug_trace":         get_events(),
    }


def build_failure_output_obj(
    record: Dict[str, Any],
    answer_type: str,
    failure_stage: str,
    debug_label: str,
    error_text: str,
    mod2_json: Optional[Dict[str, Any]] = None,
    mod3_json: Optional[Dict[str, Any]] = None,
    mod4_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Tạo một bản ghi lỗi chuẩn để không làm thất lạc sample lỗi."""
    execution = (mod4_result or {}).get("execution", {
        "status": "fail",
        "error_type": debug_label,
        "stdout": "",
        "stderr": error_text,
        "boxed": None,
        "numeric_answer": None,
        "pred_unit": "",
    })

    return {
        "id":                  record.get("id", ""),
        "detected_problem_type": record.get("_detected_problem_type",
                                             infer_problem_type(record)),
        "target_answer_type":  answer_type,
        "pipeline_status":     "fail",
        "failure_stage":       failure_stage,
        "debug_label":         debug_label,
        "question":            record.get("question", ""),
        "original_answer":     record.get("answer", ""),
        "original_unit":       record.get("unit", ""),
        "module2_extract":     mod2_json or {},
        "module3_si_extract":  mod3_json or {},
        "module4_symcode":     (mod4_result or {}).get("code", ""),
        "execution":           execution,
        "verification":        {"status": "skip", "reason": debug_label},
        "debugged":            (mod4_result or {}).get("debugged", False),
        "given_audit":         (mod4_result or {}).get("given_audit", {}),
        "given_audit_history": (mod4_result or {}).get("given_audit_history", []),
        "debug_trace":         get_events(),
    }


def build_module4_retrieval_query(
    problem_type: str,
    answer_type: str,
    question: str,
    mod2_json: Dict[str, Any],
    mod3_json: Dict[str, Any],
) -> str:
    """Build a compact retrieval query for Module 4 formula/code examples."""
    target = (mod2_json.get("target") or [{}])[0] if isinstance(mod2_json.get("target"), list) else {}
    payload = {
        "problem_type": problem_type,
        "target_answer_type": answer_type,
        "question": question,
        "target": target,
        "given": mod3_json.get("given", []),
        "conditions": mod3_json.get("conditions", []),
    }
    return json.dumps(payload, ensure_ascii=False)



# ===== Runner helpers extracted from notebook cell 35 =====
def run_pipeline_records(
    run_records: List[Dict[str, Any]],
    output_path: str,
    failed_path: str,
    report_path: str,
    run_name: str,
    resume: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Chạy toàn bộ pipeline và luôn ghi nhận cả pass lẫn fail."""
    from kaggle_pipeline import core as runtime_config

    if resume:
        outputs = read_jsonl_objects(output_path)
        failures = read_jsonl_objects(failed_path)
        completed_ids = {
            str(obj.get("id", ""))
            for obj in outputs + failures
            if str(obj.get("id", "")).strip()
        }
        if completed_ids:
            print(f"Resume enabled: loaded {len(completed_ids)} completed record IDs", flush=True)
    else:
        reset_file(output_path)
        reset_file(failed_path)
        outputs = []
        failures = []
        completed_ids = set()

    candidate_problem_types = runtime_config.bank.get("by_problem_type", {})

    for index, record in enumerate(run_records, start=1):
        record_id = record.get("id", index)
        if str(record_id) in completed_ids:
            print(f"\n[{run_name} {index}/{len(run_records)}] ID={record_id} already done; skipping", flush=True)
            continue

        question = record.get("question", "")
        problem_type = record.get("_detected_problem_type") or infer_problem_type(record)
        answer_type = "numeric_compute"
        reset_events()
        # Retrieval intentionally ignores query_id. We allow the retriever to
        # pick the closest question text, even if the same ID exists in the bank.
        exclude_ids = None

        print(f"\n{'='*60}", flush=True)
        print(f"[{run_name} {index}/{len(run_records)}] ID={record_id}", flush=True)

        try:
            if problem_type == "unknown":
                rag_problem_type, rag_info = vote_problem_type_from_rag(
                    runtime_config.bank,
                    question,
                    exclude_ids=exclude_ids,
                )
                if rag_problem_type != "unknown":
                    problem_type = rag_problem_type
                    score_ratio = rag_info.get("score_ratio", 0)
                    ratio_text = "inf" if score_ratio == float("inf") else f"{score_ratio:.2f}"
                    print(
                        "  Problem type: RAG vote "
                        f"{problem_type} (hits={rag_info.get('top_count')}, "
                        f"ratio={ratio_text})",
                        flush=True,
                    )

            if problem_type == "unknown":
                filtered_candidate_problem_types = {
                    label: [
                        ex for ex in examples
                        if not exclude_ids
                        or (
                            str(ex.get("id", "")).strip() not in exclude_ids
                            and str(ex.get("source_id", "")).strip() not in exclude_ids
                        )
                    ]
                    for label, examples in candidate_problem_types.items()
                }
                problem_type = generate_problem_type(
                    runtime_config.client,
                    runtime_config.MODEL_NAME,
                    question,
                    filtered_candidate_problem_types,
                )

            record["_detected_problem_type"] = problem_type
            print(f"  Problem type: {problem_type}", flush=True)

            fs_mod2 = select_few_shot_for_module2(
                runtime_config.bank,
                problem_type,
                runtime_config.NUM_FEW_SHOT,
                exclude_ids=exclude_ids,
                query=question,
            )
            log_json(f"{record_id}_module2_retrieved_fewshots", {
                "record_id": record_id,
                "question": question,
                "problem_type": problem_type,
                "retrieval_backend": getattr(runtime_config, "RETRIEVAL_BACKEND", "bm25"),
                "fewshot_ids": [ex.get("id") for ex in fs_mod2],
                "fewshots": [build_module2_few_shot_example(ex) for ex in fs_mod2],
            })
            print(f"  Module 2: generating extraction ({len(fs_mod2)} few-shot examples)", flush=True)
            mod2_json = generate_module2(runtime_config.client, runtime_config.MODEL_NAME, question, fs_mod2)
            if not mod2_json:
                output_obj = build_failure_output_obj(
                    record=record,
                    answer_type=answer_type,
                    failure_stage="module2_extraction",
                    debug_label="module2_invalid_or_empty",
                    error_text="Module 2 không trả về JSON hợp lệ sau số lần retry cho phép.",
                )
                append_jsonl(failed_path, output_obj)
                failures.append(output_obj)
                print("  ✗ FAIL at Module 2", flush=True)
                continue

            print("  Module 2: extraction parsed", flush=True)
            answer_type = detect_answer_type(question, mod2_json)
            mod3_json = process_module3(mod2_json)
            print(f"  Module 3: SI normalization done; answer_type={answer_type}", flush=True)

            payload = build_module4_payload(problem_type, answer_type, question, mod3_json, None)
            module4_query = build_module4_retrieval_query(
                problem_type,
                answer_type,
                question,
                mod2_json,
                mod3_json,
            )
            fs_mod4 = select_few_shot_for_module4(
                runtime_config.bank,
                answer_type,
                runtime_config.NUM_FEW_SHOT,
                exclude_ids=exclude_ids,
                query=module4_query,
                problem_type=problem_type,
            )
            log_json(f"{record_id}_module4_retrieval_and_fewshots", {
                "record_id": record_id,
                "question": question,
                "problem_type": problem_type,
                "answer_type": answer_type,
                "module2_extract": mod2_json,
                "module3_si_extract": mod3_json,
                "module4_payload": payload,
                "module4_retrieval_query": module4_query,
                "retrieval_backend": getattr(runtime_config, "RETRIEVAL_BACKEND", "bm25"),
                "fewshot_ids": [ex.get("id") for ex in fs_mod4],
                "fewshots": [build_module4_few_shot_example(ex) for ex in fs_mod4],
            })
            print(
                f"  Module 4: one-shot SymCode "
                f"({len(fs_mod4)} few-shot examples)",
                flush=True,
            )
            mod4_result = generate_and_execute_with_debug(
                runtime_config.client,
                runtime_config.MODEL_NAME,
                answer_type,
                payload,
                fs_mod4,
                runtime_config.EXEC_TIMEOUT,
                enable_debug=runtime_config.ENABLE_DEBUG,
            )

            output_obj = build_output_obj(record, answer_type, mod2_json, mod3_json, mod4_result)

            if output_obj["pipeline_status"] == "pass":
                append_jsonl(output_path, output_obj)
                outputs.append(output_obj)
                print(f"  ✓ PASS: boxed={output_obj['execution'].get('boxed')}", flush=True)
            else:
                append_jsonl(failed_path, output_obj)
                failures.append(output_obj)
                print(f"  ✗ {output_obj['pipeline_status'].upper()}: {output_obj['debug_label']}", flush=True)

        except Exception as exc:
            output_obj = build_failure_output_obj(
                record=record,
                answer_type=answer_type,
                failure_stage="unexpected_exception",
                debug_label=type(exc).__name__,
                error_text=str(exc),
            )
            append_jsonl(failed_path, output_obj)
            failures.append(output_obj)
            print(f"  ✗ ERROR: {exc}", flush=True)

    write_report(report_path, outputs, failures, len(run_records))
    print(f"{run_name} completed! Pass={len(outputs)} Fail={len(failures)}", flush=True)
    print(f"Report: {report_path}", flush=True)
    return outputs, failures
