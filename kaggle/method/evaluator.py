"""
Benchmark evaluation routines, dataset loading, metric calculations, and JSON result saving.
Includes subject x difficulty multi-dimensional metrics, SymCode verifier loop, and auto-resume checkpointing.
"""

import os
import gc
import json
import time
from typing import Dict, Any, List, Optional
try:
    import torch
except ImportError:
    torch = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .prompts import build_prompt_messages, build_retry_prompt_messages
from .extractor import (
    extract_boxed_content,
    extract_python_code,
    extract_ground_truth,
    check_exact_match
)
from .sandbox import execute_code_safely
from .verifier import verify_candidate_answer

try:
    from .model import LLMRunner
except ImportError:
    LLMRunner = Any


def parse_difficulty_level(val: Any) -> Optional[int]:
    """Parses difficulty level from integer, string (e.g. 'Level 2'), or None."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    s = str(val).lower().replace("level", "").strip()
    try:
        return int(s)
    except ValueError:
        return None


def load_dataset_file(
    file_or_hf_id: str,
    split: str = "test",
    num_samples: Optional[int] = None,
    filter_levels: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Loads dataset samples from local JSONL or HuggingFace dataset.
    Extracts question, answer, subject, and difficulty level schemas faithfully.
    """
    samples = []
    
    def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
        q = item.get("question") or item.get("problem", "")
        a = item.get("answer") or item.get("solution", "")
        
        # Extract subject
        subject = item.get("subject") or item.get("type")
        if not subject:
            # Check if GSM8K format (question + answer with ####)
            if "####" in str(a) or "question" in item:
                subject = "Grade School Math"
            else:
                subject = "unknown"
                
        # Extract level / difficulty
        lvl = parse_difficulty_level(item.get("level") or item.get("difficulty"))
        level_label = f"Level {lvl}" if lvl is not None else "N/A"
        
        return {
            "question": q,
            "answer": a,
            "subject": str(subject),
            "level": lvl,
            "level_label": level_label,
            "raw": item
        }

    if os.path.exists(file_or_hf_id):
        print(f"[*] Loading local dataset file: {file_or_hf_id}")
        with open(file_or_hf_id, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    sample = process_item(item)
                    if filter_levels is not None and sample["level"] is not None:
                        if sample["level"] not in filter_levels:
                            continue
                    samples.append(sample)
    else:
        print(f"[*] Fetching from HuggingFace dataset: {file_or_hf_id} (split: {split})...")
        from datasets import load_dataset
        ds = load_dataset(file_or_hf_id, split=split)
        for item in ds:
            sample = process_item(item)
            if filter_levels is not None and sample["level"] is not None:
                if sample["level"] not in filter_levels:
                    continue
            samples.append(sample)

    if num_samples is not None:
        samples = samples[:num_samples]
    
    if filter_levels is not None:
        print(f"[*] Filtered for Levels {filter_levels}: {len(samples)} samples found.")
    else:
        print(f"[*] Loaded {len(samples)} samples.")
    return samples


def _load_existing_checkpoint(checkpoint_file: Optional[str], method_name: str) -> Dict[str, Any]:
    """Helper to load existing results dict and results list for a method if checkpoint exists."""
    if not checkpoint_file or not os.path.exists(checkpoint_file):
        return {"data": {}, "method_results": []}
    
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            method_results = data.get("results", {}).get(method_name, [])
            return {"data": data, "method_results": method_results}
    except Exception:
        return {"data": {}, "method_results": []}


def _save_intermediate_checkpoint(
    checkpoint_file: Optional[str],
    method_name: str,
    results: List[Dict[str, Any]],
    extra_meta: Optional[Dict[str, Any]] = None
):
    """Helper to save intermediate checkpoint without corrupting other methods' results."""
    if not checkpoint_file:
        return
    
    existing_data = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = {}
            
    if "results" not in existing_data:
        existing_data["results"] = {}
    existing_data["results"][method_name] = results
    
    if extra_meta:
        for k, v in extra_meta.items():
            if k not in existing_data:
                existing_data[k] = v
                
    try:
        temp_file = checkpoint_file + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        os.replace(temp_file, checkpoint_file)
    except Exception as e:
        print(f"[!] Warning: failed to save checkpoint: {e}")


def evaluate_direct_or_cot(
    method_name: str,
    dataset: List[Dict[str, Any]],
    llm: LLMRunner,
    checkpoint_file: Optional[str] = None,
    save_every: int = 5
) -> List[Dict[str, Any]]:
    """Runs Direct or CoT zero-shot evaluation with auto-resume."""
    ckpt = _load_existing_checkpoint(checkpoint_file, method_name)
    results = ckpt["method_results"]
    completed_problems = {r["problem"] for r in results}
    
    if completed_problems:
        print(f"[*] Resuming {method_name}: {len(completed_problems)}/{len(dataset)} samples already done.")

    print(f"\n==================== Running Baseline: {method_name} ====================")
    
    new_evaluated = 0
    for item in tqdm(dataset, desc=f"Evaluating {method_name}"):
        question = item["question"]
        if question in completed_problems:
            continue
            
        gt = extract_ground_truth(item.get("raw") or item["answer"])
        messages = build_prompt_messages(method_name, question)
        
        raw_output, token_count = llm.generate_chat(messages)
        predicted_ans = extract_boxed_content(raw_output)
        is_correct = check_exact_match(predicted_ans, gt)

        results.append({
            "problem": question,
            "subject": item.get("subject", "unknown"),
            "level": item.get("level"),
            "level_label": item.get("level_label", "N/A"),
            "ground_truth": gt,
            "predicted": predicted_ans,
            "is_correct": is_correct,
            "generated_tokens": token_count,
            "attempts": 1,
            "execution_status": "not_applicable",
            "verification_status": "not_applicable",
            "verification_feedback": None,
            "raw_output": raw_output
        })
        completed_problems.add(question)
        new_evaluated += 1
        
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if checkpoint_file and (new_evaluated % save_every == 0):
            gc.collect()
            _save_intermediate_checkpoint(checkpoint_file, method_name, results)

    if checkpoint_file:
        _save_intermediate_checkpoint(checkpoint_file, method_name, results)
        
    return results


def evaluate_symcode(
    dataset: List[Dict[str, Any]],
    llm: LLMRunner,
    timeout: int = 15,
    max_retries: int = 2,
    checkpoint_file: Optional[str] = None,
    save_every: int = 5
) -> List[Dict[str, Any]]:
    """
    Runs SymCode (Neurosymbolic Equation Solving with SymPy & Verification Loop)
    with iterative feedback on execution errors and independent verification failures.
    Verifier operates 100% independently without accessing ground truth.
    """
    ckpt = _load_existing_checkpoint(checkpoint_file, "SymCode")
    results = ckpt["method_results"]
    completed_problems = {r["problem"] for r in results}
    
    if completed_problems:
        print(f"[*] Resuming SymCode: {len(completed_problems)}/{len(dataset)} samples already done.")

    print(f"\n==================== Running Method: SymCode (Max Retries: {max_retries}) ====================")
    
    new_evaluated = 0
    for item in tqdm(dataset, desc="Evaluating SymCode"):
        question = item["question"]
        if question in completed_problems:
            continue
            
        gt = extract_ground_truth(item.get("raw") or item["answer"])
        
        total_tokens = 0
        attempt = 0
        prev_code = ""
        error_tb = None
        candidate_ans = None
        verif_status = "unknown"
        verif_feedback = None
        exec_res = {}
        raw_outputs = []
        attempt_history = []
        
        while attempt <= max_retries:
            attempt += 1
            if attempt == 1:
                messages = build_prompt_messages("SymCode", question)
            else:
                messages = build_retry_prompt_messages(
                    question=question,
                    prev_code=prev_code,
                    execution_status=exec_res.get("status", "error"),
                    error_tb=error_tb,
                    candidate_answer=candidate_ans,
                    verification_status=verif_status,
                    verification_feedback=verif_feedback
                )
            
            raw_output, token_count = llm.generate_chat(messages)
            total_tokens += token_count
            raw_outputs.append(raw_output)
            
            extracted_code = extract_python_code(raw_output)
            exec_res = execute_code_safely(extracted_code, mode="symcode", timeout=timeout)
            candidate_ans = exec_res.get("extracted_answer")
            
            # Independent verification check (NO ground truth used)
            if exec_res["status"] == "success" and candidate_ans is not None:
                verif_status, verif_feedback = verify_candidate_answer(
                    question, candidate_ans, extracted_code, exec_res.get("stdout")
                )
            else:
                verif_status = "fail"
                verif_feedback = exec_res.get("traceback") or "Execution produced no output or did not print \\boxed{}."
                
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
            
            # If code executed successfully AND verification passed / is acceptable (not failed)
            if exec_res["status"] == "success" and candidate_ans is not None and verif_status != "fail":
                break
                
            # Prepare feedback for next retry attempt
            prev_code = extracted_code
            error_tb = exec_res.get("traceback")

        final_predicted = candidate_ans
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
            "execution_status": exec_res.get("status", "unknown"),
            "verification_status": verif_status,
            "verification_feedback": verif_feedback,
            "stdout": exec_res.get("stdout", ""),
            "traceback": exec_res.get("traceback"),
            "extracted_code": extracted_code,
            "raw_output": raw_outputs[-1] if raw_outputs else "",
            "raw_outputs": raw_outputs,
            "attempt_history": attempt_history
        })
        completed_problems.add(question)
        new_evaluated += 1
        
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if checkpoint_file and (new_evaluated % save_every == 0):
            gc.collect()
            _save_intermediate_checkpoint(checkpoint_file, "SymCode", results)

    if checkpoint_file:
        _save_intermediate_checkpoint(checkpoint_file, "SymCode", results)
        
    return results


def compute_metrics_table(all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics:
    1. Overall Accuracy (EM), Avg Tokens, Avg Attempts, Exec & Verif Success Rates.
    2. Accuracy breakdown by Subject.
    3. Accuracy breakdown by Difficulty Level.
    4. Accuracy breakdown by Subject x Difficulty Level.
    """
    summary_data = {}

    print("\n" + "=" * 90)
    print(f"{'OVERALL BENCHMARK SUMMARY':^90}")
    print("=" * 90)
    print(f"{'METHOD':<12} | {'ACCURACY':<10} | {'CORRECT/TOTAL':<15} | {'AVG TOKENS':<12} | {'AVG ATTEMPTS':<12} | {'EXEC RATE':<10} | {'VERIF PASS':<10}")
    print("-" * 90)

    for method, res_list in all_results.items():
        total = len(res_list)
        if total == 0:
            continue
        
        num_correct = sum(1 for r in res_list if r.get("is_correct"))
        acc = (num_correct / total) * 100.0
        avg_tokens = sum(r.get("generated_tokens", 0) for r in res_list) / total
        avg_attempts = sum(r.get("attempts", 1) for r in res_list) / total
        
        # Execution rate
        exec_applicable = [r for r in res_list if r.get("execution_status") not in ["not_applicable", None]]
        if exec_applicable:
            exec_success = sum(1 for r in exec_applicable if r.get("execution_status") == "success")
            exec_rate_str = f"{(exec_success / len(exec_applicable)) * 100.0:.1f}%"
            exec_rate_val = round((exec_success / len(exec_applicable)) * 100.0, 2)
        else:
            exec_rate_str = "N/A"
            exec_rate_val = None

        # Verification rate
        verif_applicable = [r for r in res_list if r.get("verification_status") not in ["not_applicable", None]]
        if verif_applicable:
            verif_pass = sum(1 for r in verif_applicable if r.get("verification_status") == "pass")
            verif_rate_str = f"{(verif_pass / len(verif_applicable)) * 100.0:.1f}%"
            verif_rate_val = round((verif_pass / len(verif_applicable)) * 100.0, 2)
        else:
            verif_rate_str = "N/A"
            verif_rate_val = None

        print(f"{method:<12} | {acc:<9.2f}% | {f'{num_correct}/{total}':<15} | {avg_tokens:<12.1f} | {avg_attempts:<12.2f} | {exec_rate_str:<10} | {verif_rate_str:<10}")

        # Breakdown by Subject
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

        # Breakdown by Difficulty Level
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

        # Breakdown by Subject x Difficulty
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

    # Print Subject x Difficulty Breakdown Table
    print("=" * 90)
    print(f"{'ACCURACY BREAKDOWN: SUBJECT x DIFFICULTY LEVEL':^90}")
    print("=" * 90)
    for method, s_dict in summary_data.items():
        print(f"\n--- Method: {method} ---")
        print(f"{'Subject':<28} | {'Difficulty':<12} | {'Correct/Total':<15} | {'Accuracy (%)':<15}")
        print("-" * 75)
        for key, cell in s_dict.get("by_subject_x_difficulty", {}).items():
            c_str = f"{cell['correct']}/{cell['total']}"
            print(f"{cell['subject']:<28} | {cell['difficulty']:<12} | {c_str:<15} | {cell['accuracy_percent']:<14.2f}%")

    print("=" * 90 + "\n")
    return summary_data


def save_benchmark_results(results_data: Dict[str, Any], filepath: str):
    """Saves results data dictionary to JSON file atomically."""
    temp_file = filepath + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, filepath)
    print(f"[✓] Successfully saved benchmark results to: {filepath}")
