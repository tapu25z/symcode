"""
Evaluator for SymCode (Neurosymbolic Equation Solving with SymPy & Self-Debugging Verifier).
"""

from __future__ import annotations

import os
import gc
import json
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from shared.extractor import (
    extract_boxed_content,
    extract_python_code,
    extract_ground_truth,
    check_exact_match
)
from shared.sandbox import execute_code_safely
from .verifier import verify_candidate_answer
from .prompts import build_symcode_messages, build_symcode_retry_messages


def _load_checkpoint(checkpoint_file: Optional[str]) -> tuple[dict, set]:
    if not checkpoint_file or not os.path.exists(checkpoint_file):
        return {}, set()
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", {}).get("SymCode", [])
        return data, {r["problem"] for r in results}
    except Exception:
        return {}, set()


def _save_checkpoint(checkpoint_file: Optional[str], results: List[Dict[str, Any]]) -> None:
    if not checkpoint_file:
        return
    data = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
    data.setdefault("results", {})["SymCode"] = results
    temp_file = checkpoint_file + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, checkpoint_file)


def evaluate_symcode(
    dataset: List[Dict[str, Any]],
    llm: Any,
    timeout: int = 15,
    max_retries: int = 2,
    checkpoint_file: Optional[str] = None,
    save_every: int = 5
) -> List[Dict[str, Any]]:
    """
    Danh gia phuong phap SymCode voi vong lap tu sua loi Traceback + Mathematical Verifier.
    """
    existing_data, completed = _load_checkpoint(checkpoint_file)
    results = existing_data.get("results", {}).get("SymCode", [])

    if completed:
        print(f"[INFO] Tiep tuc SymCode: da hoan thanh {len(completed)}/{len(dataset)} mau.")

    print(f"\n==================== Bat dau danh gia Phuong phap: SymCode (Max Retries: {max_retries}) ====================")

    new_count = 0
    for item in tqdm(dataset, desc="Danh gia SymCode"):
        question = item["question"]
        if question in completed:
            continue

        gt = extract_ground_truth(item.get("raw") or item["answer"])

        total_tokens = 0
        attempt = 0
        prev_code = ""
        error_tb = None
        candidate_ans = None
        verif_status = "unknown"
        verif_feedback = None
        exec_res: Dict[str, Any] = {}
        raw_outputs = []
        attempt_history = []

        while attempt <= max_retries:
            attempt += 1
            if attempt == 1:
                messages = build_symcode_messages(question)
                raw_output, token_count = llm.generate_chat(messages)
            else:
                messages = build_symcode_retry_messages(
                    question=question,
                    prev_code=prev_code,
                    execution_status=exec_res.get("status", "error"),
                    error_tb=error_tb,
                    candidate_answer=candidate_ans,
                    verification_status=verif_status,
                    verification_feedback=verif_feedback
                )
                raw_output, token_count = llm.generate_chat(messages, enable_thinking=False)

            total_tokens += token_count
            raw_outputs.append(raw_output)

            extracted_code = extract_python_code(raw_output)
            exec_res = execute_code_safely(extracted_code, mode="symcode", timeout=timeout)
            candidate_ans = exec_res.get("extracted_answer")

            if candidate_ans is not None and str(candidate_ans).strip().lower() in ["none", "null", "invalid", "undefined", "nan"]:
                candidate_ans = None

            # Kiem chung doc lap khong dung ground truth
            if exec_res.get("status") == "success" and candidate_ans is not None:
                verif_status, verif_feedback = verify_candidate_answer(
                    question, candidate_ans, extracted_code, exec_res.get("stdout")
                )
            else:
                verif_status = "fail"
                verif_feedback = exec_res.get("traceback") or "Khong in ra dap an dinh dang \\boxed{}."

            attempt_record = {
                "attempt": attempt,
                "code": extracted_code,
                "generated_tokens": token_count,
                "execution_status": exec_res.get("status"),
                "candidate_answer": candidate_ans,
                "verification_status": verif_status,
                "verification_feedback": verif_feedback,
                "stdout": exec_res.get("stdout", ""),
                "traceback": exec_res.get("traceback")
            }
            attempt_history.append(attempt_record)

            if exec_res.get("status") == "success" and candidate_ans is not None and verif_status != "fail":
                break

            prev_code = extracted_code
            error_tb = exec_res.get("traceback")

        final_predicted = candidate_ans
        if final_predicted is None or str(final_predicted).strip().lower() in ["none", "null", "invalid", "undefined", "nan"]:
            for out in reversed(raw_outputs):
                b = extract_boxed_content(out)
                if b is not None and b.strip().lower() not in ["none", "null", "invalid", "undefined", "nan"]:
                    final_predicted = b
                    break

        is_correct = check_exact_match(final_predicted, gt)

        results.append({
            "problem": question,
            "subject": item.get("subject", "unknown"),
            "level": item.get("level"),
            "level_label": item.get("level_label", "N/A"),
            "ground_truth": gt,
            "predicted": final_predicted,
            "is_correct": is_correct,
            "generated_tokens": total_tokens,
            "attempts": attempt,
            "execution_status": exec_res.get("status", "error"),
            "verification_status": verif_status,
            "verification_feedback": verif_feedback,
            "stdout": exec_res.get("stdout", ""),
            "traceback": exec_res.get("traceback"),
            "extracted_code": extracted_code,
            "raw_output": raw_outputs[-1] if raw_outputs else "",
            "raw_outputs": raw_outputs,
            "attempt_history": attempt_history
        })
        completed.add(question)
        new_count += 1

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if checkpoint_file and (new_count % max(1, save_every) == 0):
            gc.collect()
            _save_checkpoint(checkpoint_file, results)

    if checkpoint_file:
        _save_checkpoint(checkpoint_file, results)

    return results
