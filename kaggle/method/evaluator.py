"""
Module đánh giá benchmark, nạp tập dữ liệu, tính toán các chỉ số đa chiều và lưu kết quả JSON.
Bao gồm phân rã Subject x Difficulty, vòng lặp tự sửa lỗi SymCode và cơ chế tự động lưu/tiếp tục phiên (Auto-Resume).
"""

import os
import gc
import json
import re
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

from .prompts import (
    build_prompt_messages,
    build_retry_prompt_messages,
    build_symplanner_retry_prompt_messages,
    build_planner_messages,
    clean_planner_note,
    build_symplanner_codegen_messages,
    build_symplanner_debug_messages
)
from .extractor import (
    extract_boxed_content,
    extract_answer_fallback,
    extract_python_code,
    extract_symplanner_code,
    extract_ground_truth,
    check_exact_match
)
from .sandbox import execute_code_safely
from .verifier import verify_candidate_answer
from .static_lint import lint_sympy_code
from .target_contract import infer_target_spec, parse_planner_contract, format_answer_for_contract

try:
    from .model import LLMRunner
except ImportError:
    LLMRunner = Any


def _verification_rank(status: str) -> int:
    return {"pass": 2, "unknown": 1, "fail": 0}.get(str(status), -1)


def _should_retry_symplanner(execution_status: str, candidate: Any, verification_status: str, feedback: Any) -> bool:
    if execution_status != "success" or candidate is None:
        return True
    feedback_text = str(feedback or "").lower()
    actionable_tokens = (
        "invalid token",
        "unresolved free symbol",
        "target requires",
        "target must remain symbolic",
        "base notation",
        "coordinate tuple",
        "non-negative count",
        "integer count",
        "probability must",
        "undefined or infinite",
        "empty candidate",
        "did not print",
        "no candidate",
    )
    if verification_status == "fail":
        return any(token in feedback_text for token in actionable_tokens)
    # Generic unknown means the verifier cannot prove the answer without ground
    # truth. Retrying those cases usually burns tokens and can damage a good
    # candidate, so retry only concrete output-contract issues.
    return verification_status == "unknown" and any(token in feedback_text for token in ("target requires", "base notation", "coordinate tuple"))


def _with_static_diagnostics(feedback: Any, code: str, execution_status: str, candidate: Any, verification_status: str) -> Any:
    findings = lint_sympy_code(code)
    if not findings or (execution_status == "success" and candidate is not None and verification_status != "fail"):
        return feedback
    suffix = "Static diagnostics: " + "; ".join(findings)
    return f"{feedback or 'No verifier feedback.'} {suffix}"


def _candidate_is_present(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "none", "null", "invalid", "undefined", "nan"}


def _symplanner_record_rank(record: Dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        _verification_rank(record.get("verification_status")),
        int(record.get("execution_status") == "success"),
        int(_candidate_is_present(record.get("candidate_answer"))),
        int(record.get("attempt", 0)),
    )


def _translate_legacy_verification_feedback(feedback: Any) -> Any:
    """Translate legacy Vietnamese verifier feedback to English for resumed checkpoints."""
    if not isinstance(feedback, str):
        return feedback

    replacements = {
        "Lỗi kiểm chứng: Không tìm thấy đáp án ứng viên hoặc mã nguồn không in ra định dạng \\boxed{...}.": (
            "Verification Error: No candidate answer was found, or the code did not print a \\boxed{...} result."
        ),
        "Lỗi kiểm chứng: Cặp tọa độ rỗng.": "Verification Error: Empty coordinate tuple.",
        "Kiểm chứng thành công: Thỏa mãn ràng buộc tọa độ cực (r > 0 và 0 <= theta < 2*pi).": (
            "Verification Passed: Candidate answer satisfies polar-coordinate constraints (r > 0 and 0 <= theta < 2*pi)."
        ),
        "Cặp tọa độ cú pháp chuẩn.": "Candidate coordinate tuple is well-formed.",
        "Kiểm chứng thành công: Đáp án thỏa mãn khoảng xác suất [0, 1].": (
            "Verification Passed: Candidate answer satisfies probability bounds [0, 1]."
        ),
        "Đáp án hợp thức về mặt cú pháp nhưng đặc thù bài toán yêu cầu so khớp kết quả.": (
            "Candidate answer is syntactically well-formed, but problem nature prevents automated symbolic proof without ground truth."
        ),
        "Mã nguồn không in ra kết quả định dạng \\boxed{}.": "Code did not print a \\boxed{} result.",
        "Mã nguồn không in ra kết quả định dạng \\boxed{} hợp lệ.": "Code did not print a valid \\boxed{} result.",
        "Mã nguồn sửa đổi không in ra kết quả định dạng \\boxed{}.": "Repaired code did not print a \\boxed{} result.",
    }
    if feedback in replacements:
        return replacements[feedback]

    patterns = [
        (
            r"^Kiểm chứng thành công: Đáp án (.+) là số nguyên không âm hợp lệ\.$",
            r"Verification Passed: Candidate answer \1 is a valid non-negative integer count.",
        ),
        (
            r"^Kiểm chứng thành công: Đại lượng hình học có giá trị dương \((.+)\)\.$",
            r"Verification Passed: Geometric dimension is positive (\1).",
        ),
        (
            r"^Đáp án là thực thể văn bản hợp lệ \('(.+)'\)\.$",
            r"Candidate answer is a valid text entity ('\1').",
        ),
        (
            r"^Lỗi kiểm chứng: Đáp án '(.+)' là token không hợp lệ \(None/Invalid/NaN/Error/Function Object\)\. Hãy tính toán ra giá trị cụ thể\.$",
            r"Verification Error: Candidate answer '\1' is an invalid token (None/Invalid/NaN/Error/Function Object). Compute and print a concrete value.",
        ),
        (
            r"^Lỗi kiểm chứng: Đáp án '(.+)' chứa token không hợp lệ \(None/NaN/Error/Function Object\)\.$",
            r"Verification Error: Candidate answer '\1' is an invalid token (None/NaN/Error/Function Object). Compute and print a concrete value.",
        ),
        (
            r"^Lỗi kiểm chứng: Đáp án '(.+)' là tên biến Python chưa được đánh giá thành giá trị cụ thể\. Hãy tính toán giá trị của biến trước khi in\.$",
            r"Verification Error: Candidate answer '\1' is an unevaluated Python variable name. Evaluate the variable before printing it.",
        ),
        (
            r"^Lỗi kiểm chứng: Bán kính cực r phải dương \(r > 0\), nhưng nhận được r = (.+)\.$",
            r"Verification Error: The polar radius r must be positive (r > 0), but got r = \1.",
        ),
        (
            r"^Lỗi kiểm chứng: Góc cực theta phải thỏa mãn 0 <= theta < 2\*pi, nhưng nhận được theta = (.+)\.$",
            r"Verification Error: The polar angle theta must satisfy 0 <= theta < 2*pi, but got theta = \1.",
        ),
        (
            r"^Lỗi kiểm chứng: Đáp án đánh giá thành giá trị không xác định hoặc vô cực \((.+)\)\.$",
            r"Verification Error: Candidate answer evaluates to an undefined or infinite value (\1).",
        ),
        (
            r"^Lỗi kiểm chứng: Đáp án '(.+)' vẫn còn chứa biến tự do \((.+)\) chưa được giải thành số cụ thể\. Hãy giải hệ phương trình hoặc tính toán tuần tự để tìm giá trị số cụ thể\.$",
            r"Verification Error: Candidate answer '\1' still contains unresolved free symbol(s) (\2). Solve the equations or compute the result step by step to obtain a concrete numeric value.",
        ),
        (
            r"^Lỗi kiểm chứng: Bài toán yêu cầu số đếm không âm, nhưng kết quả nhận được là số âm \((.+)\)\.$",
            r"Verification Error: This problem requires a non-negative count, but the result is negative (\1).",
        ),
        (
            r"^Lỗi kiểm chứng: Bài toán yêu cầu số đếm nguyên, nhưng kết quả nhận được không phải số nguyên \((.+)\)\.$",
            r"Verification Error: This problem requires an integer count, but the result is not an integer (\1).",
        ),
        (
            r"^Lỗi kiểm chứng: Xác suất phải nằm trong đoạn \[0, 1\], nhưng kết quả là (.+)\.$",
            r"Verification Error: Probability must lie in [0, 1], but the result is \1.",
        ),
        (
            r"^Lỗi kiểm chứng: Đại lượng hình học \(độ dài/diện tích/chu vi\) phải dương, nhưng nhận được (.+)\.$",
            r"Verification Error: Geometric quantity (length/area/perimeter) must be positive, but got \1.",
        ),
    ]
    for pattern, replacement in patterns:
        if re.match(pattern, feedback):
            return re.sub(pattern, replacement, feedback)
    return feedback


def _normalize_legacy_verification_feedback(value: Any) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "verification_feedback":
                value[key] = _translate_legacy_verification_feedback(item)
            else:
                value[key] = _normalize_legacy_verification_feedback(item)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _normalize_legacy_verification_feedback(item)
    return value


def parse_difficulty_level(val: Any) -> Optional[int]:
    """Chuyển đổi mức độ khó từ số nguyên, chuỗi (ví dụ: 'Level 2'), hoặc None."""
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
    filter_levels: Optional[List[int]] = None,
    tail: bool = False,
    per_level_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Nạp dữ liệu bài toán từ file JSONL cục bộ hoặc Hugging Face dataset.
    Trích xuất câu hỏi, đáp án chuẩn, chủ đề (subject) và độ khó (difficulty level).
    Hỗ trợ trích xuất N mẫu cuối cùng (tail=True) hoặc N mẫu cho từng level (per_level_samples).
    """
    samples = []
    
    def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
        q = item.get("question") or item.get("problem", "")
        a = item.get("answer") or item.get("solution", "")
        
        subject = item.get("subject") or item.get("type")
        if not subject:
            if "####" in str(a) or "question" in item:
                subject = "Grade School Math"
            else:
                subject = "unknown"
                
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
        print(f"[INFO] Nap file du lieu cuc bo: {file_or_hf_id}")
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
        print(f"[INFO] Tai tap du lieu tu Hugging Face: {file_or_hf_id} (split: {split})...")
        from datasets import load_dataset
        ds = load_dataset(file_or_hf_id, split=split)
        for item in ds:
            sample = process_item(item)
            if filter_levels is not None and sample["level"] is not None:
                if sample["level"] not in filter_levels:
                    continue
            samples.append(sample)

    # Xử lý cắt mẫu theo per_level_samples hoặc num_samples (hỗ trợ cả mode tail - lấy mẫu từ cuối lên)
    if per_level_samples is not None:
        level_groups: Dict[Any, List[Dict[str, Any]]] = {}
        for sample in samples:
            lvl = sample.get("level")
            if lvl not in level_groups:
                level_groups[lvl] = []
            level_groups[lvl].append(sample)
        
        selected_samples = []
        for lvl in sorted(level_groups.keys(), key=lambda x: (x is None, x)):
            group = level_groups[lvl]
            if tail:
                selected_samples.extend(group[-per_level_samples:])
            else:
                selected_samples.extend(group[:per_level_samples])
        samples = selected_samples
        print(f"[INFO] Da trich xuat N mau moi level (per_level_samples={per_level_samples}, tail={tail}): Tong {len(samples)} mau.")
    elif num_samples is not None:
        if tail:
            samples = samples[-num_samples:]
            print(f"[INFO] Da lay {len(samples)} mau CUOI CUNG (tail=True).")
        else:
            samples = samples[:num_samples]
            print(f"[INFO] Da lay {len(samples)} mau DAU TIEN (tail=False).")

    if filter_levels is not None:
        print(f"[INFO] Da loc cho cac muc do {filter_levels}: {len(samples)} mau hop le.")
    else:
        print(f"[INFO] Tong so mau da nap: {len(samples)}")
    return samples


def _load_existing_checkpoint(checkpoint_file: Optional[str], method_name: str) -> Dict[str, Any]:

    
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            _normalize_legacy_verification_feedback(data)
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
    """Lưu checkpoint trung gian bằng cơ chế ghi ngầm nguyên tử (atomic write)."""
    if not checkpoint_file:
        return
    
    existing_data = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                _normalize_legacy_verification_feedback(existing_data)
        except Exception:
            existing_data = {}
            
    if "results" not in existing_data:
        existing_data["results"] = {}
    existing_data["results"][method_name] = _normalize_legacy_verification_feedback(results)
    
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
        print(f"[WARN] Khong the ghi checkpoint trung gian: {e}")


def evaluate_direct_or_cot(
    method_name: str,
    dataset: List[Dict[str, Any]],
    llm: LLMRunner,
    checkpoint_file: Optional[str] = None,
    save_every: int = 5
) -> List[Dict[str, Any]]:
    """
    Thực thi đánh giá zero-shot cho Direct hoặc Chain-of-Thought (CoT) với tính năng Auto-Resume.
    """
    ckpt = _load_existing_checkpoint(checkpoint_file, method_name)
    results = ckpt["method_results"]
    completed_problems = {r["problem"] for r in results}
    
    if completed_problems:
        print(f"[INFO] Tiep tuc phuong phap {method_name}: da hoan thanh {len(completed_problems)}/{len(dataset)} mau.")

    print(f"\n==================== Bat dau danh gia Baseline: {method_name} ====================")
    
    new_evaluated = 0
    for item in tqdm(dataset, desc=f"Danh gia {method_name}"):
        question = item["question"]
        if question in completed_problems:
            continue
            
        gt = extract_ground_truth(item.get("raw") or item["answer"])
        messages = build_prompt_messages(method_name, question)
        
        raw_output, token_count = llm.generate_chat(messages)
        predicted_ans = extract_answer_fallback(raw_output)
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
    Thực thi đánh giá phương pháp SymCode (Neurosymbolic Equation Solving với SymPy & Vòng lặp Verifier).
    Cung cấp phản hồi lỗi thực thi (Traceback) và chẩn đoán toán học độc lập (Verifier Diagnosis).
    Bộ kiểm chứng hoạt động độc lập 100%, không sử dụng ground truth.
    """
    ckpt = _load_existing_checkpoint(checkpoint_file, "SymCode")
    results = ckpt["method_results"]
    completed_problems = {r["problem"] for r in results}
    
    if completed_problems:
        print(f"[INFO] Tiep tuc phuong phap SymCode: da hoan thanh {len(completed_problems)}/{len(dataset)} mau.")

    print(f"\n==================== Bat dau danh gia Phuong phap: SymCode (So lan retry toi da: {max_retries}) ====================")
    
    new_evaluated = 0
    for item in tqdm(dataset, desc="Danh gia SymCode"):
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
                raw_output, token_count = llm.generate_chat(messages)
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
                raw_output, token_count = llm.generate_chat(messages, enable_thinking=False)
            
            total_tokens += token_count
            raw_outputs.append(raw_output)
            
            extracted_code = extract_python_code(raw_output)
            exec_res = execute_code_safely(extracted_code, mode="symcode", timeout=timeout)
            candidate_ans = exec_res.get("extracted_answer")
            
            if candidate_ans is not None and str(candidate_ans).strip().lower() in ["none", "null", "invalid", "undefined", "nan"]:
                candidate_ans = None

            # Kiểm chứng độc lập không dùng ground truth
            if exec_res["status"] == "success" and candidate_ans is not None:
                verif_status, verif_feedback = verify_candidate_answer(
                    question, candidate_ans, extracted_code, exec_res.get("stdout")
                )
            else:
                verif_status = "fail"
                verif_feedback = exec_res.get("traceback") or "Code did not print a \\boxed{} result."
                
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
            
            # Dừng nếu code chạy thành công VÀ vượt qua kiểm chứng (không bị fail)
            if exec_res["status"] == "success" and candidate_ans is not None and verif_status != "fail":
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


def evaluate_symplanner(
    dataset: List[Dict[str, Any]],
    llm: LLMRunner,
    timeout: int = 15,
    max_retries: int = 2,
    checkpoint_file: Optional[str] = None,
    save_every: int = 5
) -> List[Dict[str, Any]]:
    """
    Thực thi đánh giá phương pháp SymPlanner theo kiến trúc Decoupled Multi-Turn Pipeline:
    - Turn 1 (Planner Phase): Phân tích & Lập kế hoạch ngắn gọn (<120 từ / JSON), không sinh code.
    - Turn 2 (Pure Codegen Phase): Nhận Đề bài + Kế hoạch, sinh 100% mã nguồn Python/SymPy thực thi thuần túy.
    - Turn 3 (Execution & Verifier): Chạy trong Sandbox, kiểm chứng toán học độc lập (không dùng ground truth).
    - Turn 4 (Targeted Debug Repair): Sửa lỗi trực tiếp vào code nếu gặp lỗi Runtime/Cú pháp/Free Symbols/Verifier.
    """
    ckpt = _load_existing_checkpoint(checkpoint_file, "SymPlanner")
    results = ckpt["method_results"]
    completed_problems = {r["problem"] for r in results}
    
    if completed_problems:
        print(f"[INFO] Tiep tuc phuong phap SymPlanner: da hoan thanh {len(completed_problems)}/{len(dataset)} mau.")

    print(f"\n==================== Bat dau danh gia Phuong phap: SymPlanner (Decoupled Multi-Turn Pipeline, Retries: {max_retries}) ====================")
    
    new_evaluated = 0
    for item in tqdm(dataset, desc="Danh gia SymPlanner"):
        question = item["question"]
        if question in completed_problems:
            continue
            
        gt = extract_ground_truth(item.get("raw") or item["answer"])
        
        total_tokens = 0
        attempt_history = []
        raw_outputs = []
        
        # -------------------------------------------------------------
        # TURN 1: PLANNER PHASE (Chỉ lập kế hoạch ngắn gọn, không sinh code)
        # -------------------------------------------------------------
        planner_messages = build_planner_messages(question)
        raw_plan, plan_tokens = llm.generate_chat(planner_messages, max_new_tokens_override=384)
        total_tokens += plan_tokens
        raw_outputs.append(f"### Turn 1 (Planner Note):\n{raw_plan}")
        planner_note, planner_meta, planner_errors = parse_planner_contract(raw_plan, question)
        if planner_errors:
            # Do not feed truncated/non-JSON planner text into codegen. Preserve
            # the raw response for diagnostics and pass a bounded safe fallback.
            fallback_spec = infer_target_spec(question, planner_note)
            planner_note = json.dumps({
                "target_unknown": "infer from problem",
                "given_constants": [],
                "strategy": "solve directly from the problem",
                "steps": [],
                "pitfalls": planner_errors,
                "answer_type": fallback_spec["answer_type"],
            }, ensure_ascii=True)
            planner_meta = {"answer_type": fallback_spec["answer_type"]}

        # -------------------------------------------------------------
        # TURN 2: PURE CODEGEN PHASE (Sinh 100% Python/SymPy code)
        # -------------------------------------------------------------
        codegen_messages = build_symplanner_codegen_messages(question, planner_note)
        raw_code_output, code_tokens = llm.generate_chat(codegen_messages, enable_thinking=False)
        total_tokens += code_tokens
        raw_outputs.append(f"### Turn 2 (Codegen Initial):\n{raw_code_output}")
        
        extracted_code = extract_python_code(raw_code_output)
        exec_res = execute_code_safely(extracted_code, mode="symplanner", timeout=timeout)
        candidate_ans = exec_res.get("extracted_answer")
        
        if candidate_ans is not None and str(candidate_ans).strip().lower() in ["none", "null", "invalid", "undefined", "nan"]:
            candidate_ans = None

        if exec_res["status"] == "success" and candidate_ans is not None:
            verif_status, verif_feedback = verify_candidate_answer(
                question, candidate_ans, extracted_code, exec_res.get("stdout"), planner_note
            )
        else:
            verif_status = "fail"
            verif_feedback = exec_res.get("traceback") or "Code did not print a valid \\boxed{} result."
        verif_feedback = _with_static_diagnostics(
            verif_feedback, extracted_code, exec_res.get("status", "error"), candidate_ans, verif_status
        )
            
        attempt_record = {
            "attempt": 1,
            "phase": "codegen",
            "code": extracted_code,
            "generated_tokens": code_tokens,
            "execution_status": exec_res.get("status"),
            "candidate_answer": candidate_ans,
            "canonical_answer": exec_res.get("canonical_answer"),
            "answer_type": exec_res.get("answer_type"),
            "unit": exec_res.get("unit"),
            "variables": exec_res.get("variables"),
            "verification_status": verif_status,
            "verification_feedback": verif_feedback,
            "static_lint": lint_sympy_code(extracted_code),
            "stdout": exec_res.get("stdout", ""),
            "traceback": exec_res.get("traceback")
        }
        attempt_history.append(attempt_record)

        attempt = 1
        # -------------------------------------------------------------
        # TURN 3: TARGETED DEBUG REPAIR LOOP (Nếu chạy lỗi hoặc Verifier fail)
        # -------------------------------------------------------------
        while attempt <= max_retries and _should_retry_symplanner(exec_res.get("status", "error"), candidate_ans, verif_status, verif_feedback):
            attempt += 1
            if sum(1 for record in attempt_history if record.get("code") == extracted_code) >= 2:
                verif_feedback = f"{verif_feedback or 'No actionable diagnosis.'} Previous repair repeated the same code; produce a materially different implementation."
            debug_messages = build_symplanner_debug_messages(
                question=question,
                bad_code=extracted_code,
                execution_status=exec_res.get("status", "error"),
                error_tb=exec_res.get("traceback"),
                candidate_answer=candidate_ans,
                verification_status=verif_status,
                verification_feedback=verif_feedback,
                planner_note=planner_note
            )
            raw_debug_output, dbg_tokens = llm.generate_chat(debug_messages, enable_thinking=False)
            total_tokens += dbg_tokens
            raw_outputs.append(f"### Turn 3 (Debug Retry {attempt}):\n{raw_debug_output}")
            
            extracted_code = extract_python_code(raw_debug_output)
            exec_res = execute_code_safely(extracted_code, mode="symplanner", timeout=timeout)
            candidate_ans = exec_res.get("extracted_answer")
            
            if candidate_ans is not None and str(candidate_ans).strip().lower() in ["none", "null", "invalid", "undefined", "nan"]:
                candidate_ans = None

            if exec_res["status"] == "success" and candidate_ans is not None:
                verif_status, verif_feedback = verify_candidate_answer(
                    question, candidate_ans, extracted_code, exec_res.get("stdout"), planner_note
                )
            else:
                verif_status = "fail"
                verif_feedback = exec_res.get("traceback") or "Repaired code did not print a \\boxed{} result."
            verif_feedback = _with_static_diagnostics(
                verif_feedback, extracted_code, exec_res.get("status", "error"), candidate_ans, verif_status
            )
                
            retry_record = {
                "attempt": attempt,
                "phase": "debug_repair",
                "code": extracted_code,
                "generated_tokens": dbg_tokens,
                "execution_status": exec_res.get("status"),
                "candidate_answer": candidate_ans,
                "canonical_answer": exec_res.get("canonical_answer"),
                "answer_type": exec_res.get("answer_type"),
                "unit": exec_res.get("unit"),
                "variables": exec_res.get("variables"),
                "verification_status": verif_status,
                "verification_feedback": verif_feedback,
                "static_lint": lint_sympy_code(extracted_code),
                "stdout": exec_res.get("stdout", ""),
                "traceback": exec_res.get("traceback")
            }
            attempt_history.append(retry_record)

            if sum(1 for record in attempt_history if record.get("code") == extracted_code) >= 2:
                break
            
            if not _should_retry_symplanner(exec_res.get("status", "error"), candidate_ans, verif_status, verif_feedback):
                break

        # -------------------------------------------------------------
        # FINAL ANSWER EXTRACTION & ACCURACY EVALUATION
        # -------------------------------------------------------------
        # Keep the strongest non-failing attempt. A repair response can execute
        # successfully while silently degrading a previously valid answer.
        best_record = max(
            attempt_history,
            key=_symplanner_record_rank,
            default={}
        )
        final_predicted = best_record.get("candidate_answer", candidate_ans)
        final_exec_status = best_record.get("execution_status", exec_res.get("status", "unknown"))
        final_verif_status = best_record.get("verification_status", verif_status)
        final_verif_feedback = best_record.get("verification_feedback", verif_feedback)
        final_stdout = best_record.get("stdout", exec_res.get("stdout", ""))
        final_traceback = best_record.get("traceback", exec_res.get("traceback"))
        final_code = best_record.get("code", extracted_code)
        final_canonical = best_record.get("canonical_answer")
        final_answer_type = best_record.get("answer_type")
        final_unit = best_record.get("unit")
        if final_predicted is None or str(final_predicted).strip().lower() in ["none", "null", "invalid", "undefined", "nan"]:
            # Fallback an toàn: trích xuất từ planner note nếu có
            box_match = extract_boxed_content(planner_note)
            if box_match:
                final_predicted = box_match
        final_predicted = format_answer_for_contract(question, final_predicted, final_answer_type)

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
            "execution_status": final_exec_status,
            "verification_status": final_verif_status,
            "verification_feedback": final_verif_feedback,
            "stdout": final_stdout,
            "traceback": final_traceback,
            "extracted_code": final_code,
            "canonical_answer": final_canonical,
            "answer_type": final_answer_type,
            "unit": final_unit,
            "planner_note": planner_note,
            "planner_contract": planner_meta,
            "planner_errors": planner_errors,
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
            _save_intermediate_checkpoint(checkpoint_file, "SymPlanner", results)

    if checkpoint_file:
        _save_intermediate_checkpoint(checkpoint_file, "SymPlanner", results)
        
    return results



def compute_metrics_table(all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Tính toán bảng tổng hợp chỉ số đánh giá toàn diện:
    1. Overall: Accuracy (Exact Match), Tokens trung bình, Số lần thử (Attempts), Tỷ lệ thực thi & kiểm chứng.
    2. Phân rã theo Chủ đề (by Subject).
    3. Phân rã theo Mức độ khó (by Difficulty Level).
    4. Phân rã đa chiều theo Subject x Difficulty Level.
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
        normalized_correct = sum(
            1 for r in res_list
            if check_exact_match(
                format_answer_for_contract(r.get("problem", ""), r.get("predicted"), r.get("answer_type")),
                r.get("ground_truth", "")
            )
        )
        acc = (num_correct / total) * 100.0
        avg_tokens = sum(r.get("generated_tokens", 0) for r in res_list) / total
        avg_attempts = sum(r.get("attempts", 1) for r in res_list) / total
        
        # Tỷ lệ thực thi thành công
        exec_applicable = [r for r in res_list if r.get("execution_status") not in ["not_applicable", None]]
        if exec_applicable:
            exec_success = sum(1 for r in exec_applicable if r.get("execution_status") == "success")
            exec_rate_str = f"{(exec_success / len(exec_applicable)) * 100.0:.1f}%"
        else:
            exec_rate_str = "N/A"

        # Tỷ lệ vượt qua kiểm chứng độc lập
        verif_applicable = [r for r in res_list if r.get("verification_status") not in ["not_applicable", None]]
        if verif_applicable:
            verif_pass = sum(1 for r in verif_applicable if r.get("verification_status") == "pass")
            verif_rate_str = f"{(verif_pass / len(verif_applicable)) * 100.0:.1f}%"
        else:
            verif_rate_str = "N/A"

        false_passes = sum(1 for r in res_list if r.get("verification_status") == "pass" and not r.get("is_correct"))
        false_fails = sum(1 for r in res_list if r.get("verification_status") == "fail" and r.get("is_correct"))
        malformed_plans = sum(1 for r in res_list if r.get("planner_errors"))
        recovered = 0
        for record in res_list:
            history = record.get("attempt_history") or []
            if len(history) > 1 and history[0].get("candidate_answer") is not None and record.get("is_correct"):
                first_pred = format_answer_for_contract(
                    record.get("problem", ""),
                    history[0].get("candidate_answer"),
                    history[0].get("answer_type"),
                )
                if not check_exact_match(first_pred, record.get("ground_truth", "")):
                    recovered += 1

        print(f"{method:<14} | {acc:<9.2f}% | {f'{num_correct}/{total}':<15} | {avg_tokens:<12.1f} | {avg_attempts:<12.2f} | {exec_rate_str:<10} | {verif_rate_str:<10}")

        # Phân rã theo Subject
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

        # Phân rã theo Difficulty Level
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

        # Phân rã theo Subject x Difficulty
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
            "normalized_accuracy_percent": round((normalized_correct / total) * 100.0, 2),
            "normalized_match_count": normalized_correct,
            "exact_match_count": num_correct,
            "total_samples": total,
            "avg_generated_tokens": round(avg_tokens, 1),
            "avg_attempts": round(avg_attempts, 2),
            "execution_success_rate": exec_rate_str,
            "verification_success_rate": verif_rate_str,
            "verification_false_passes": false_passes,
            "verification_false_fails": false_fails,
            "planner_malformed_count": malformed_plans,
            "repair_recovered_count": recovered,
            "by_subject": by_subject,
            "by_difficulty": by_difficulty,
            "by_subject_x_difficulty": by_subject_x_difficulty
        }

    print("=" * 90 + "\n")

    # In bảng chi tiết Subject x Difficulty Level
    print("=" * 90)
    print(f"{'CHI TIET ACCURACY: SUBJECT x DIFFICULTY LEVEL':^90}")
    print("=" * 90)
    for method, s_dict in summary_data.items():
        print(f"\n--- Phuong phap: {method} ---")
        print(f"{'Chu de (Subject)':<28} | {'Do kho':<12} | {'Dung/Tong':<15} | {'Accuracy (%)':<15}")
        print("-" * 75)
        for key, cell in s_dict.get("by_subject_x_difficulty", {}).items():
            c_str = f"{cell['correct']}/{cell['total']}"
            print(f"{cell['subject']:<28} | {cell['difficulty']:<12} | {c_str:<15} | {cell['accuracy_percent']:<14.2f}%")

    print("=" * 90 + "\n")
    return summary_data


def save_benchmark_results(results_data: Dict[str, Any], filepath: str):
    """Lưu dữ liệu kết quả benchmark vào file JSON bằng phương thức an toàn."""
    temp_file = filepath + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, filepath)
    print(f"[SUCCESS] Da luu ket qua benchmark thanh cong tai: {filepath}")
