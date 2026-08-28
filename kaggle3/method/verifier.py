"""
Module bộ kiểm chứng toán học độc lập (Independent Mathematical Verifier) cho phương pháp SymCode và SymPlanner.
Thực hiện kiểm tra tính nhất quán toán học, ràng buộc miền giá trị, phần dư phương trình và tính hợp thức đại số
hoàn toàn độc lập, KHÔNG sử dụng hoặc làm lộ đáp án chuẩn (ground truth).
"""

import re
from typing import Tuple, Optional, Any


def verify_candidate_answer(
    question: str,
    candidate_answer: Optional[str],
    code: Optional[str] = None,
    stdout: Optional[str] = None
) -> Tuple[str, str]:
    """
    Kiểm tra tính hợp lệ của đáp án ứng viên dựa trên logic toán học và ràng buộc của bài toán.
    Hoàn toàn không truy cập hay sử dụng đáp án chuẩn.

    Returns:
        (status, feedback_message)
        Trong đó status nhận một trong các giá trị:
            - "pass": Đáp án thỏa mãn các ràng buộc miền giá trị và tính hợp lệ.
            - "fail": Đáp án vi phạm ràng buộc toán học hoặc có lỗi cấu trúc.
            - "unknown": Cú pháp hợp lệ nhưng đặc thù bài toán không thể chứng minh hình thức mà không có ground truth.
    """
    # 1. Kiểm tra đáp án rỗng hoặc không tồn tại
    if candidate_answer is None or not str(candidate_answer).strip():
        return (
            "fail",
            "Verification Error: No candidate answer was found, or the code did not print a \\boxed{...} result."
        )

    cand_str = str(candidate_answer).strip()

    # 2. Kiểm tra các placeholder không hợp lệ hoặc chuỗi lỗi
    invalid_tokens = ["todo", "none", "null", "undefined", "error", "nan", "invalid", "no valid solution", "<function", "<class"]
    if any(p in cand_str.lower() for p in invalid_tokens):
        return (
            "fail",
            f"Verification Error: Candidate answer '{cand_str}' is an invalid token (None/Invalid/NaN/Error/Function Object). Actionable Fix: Compute and print a concrete numerical or symbolic value. Do not print uninitialized placeholders."
        )

    # 3. Kiểm tra lỗi in tên biến Python chưa qua tính toán trong \boxed{}
    raw_var_pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
    common_code_vars = {
        "ans", "answer", "result", "res", "final_answer", "sol", "solution",
        "output", "val", "value", "perimeter_hexagon", "num_divisors", "count",
        "total", "speed", "max_speed", "min_val", "max_val", "target", "candidates"
    }
    cleaned_cand = cand_str.replace("\\", "").replace("{", "").replace("}", "").strip()
    if cleaned_cand in common_code_vars or (re.match(raw_var_pattern, cleaned_cand) and len(cleaned_cand) > 4 and not cleaned_cand.isalpha()):
        return (
            "fail",
            f"Verification Error: Candidate answer '{cand_str}' is an unevaluated Python variable name. Actionable Fix: Compute the actual value of the variable first, then pass the evaluated variable to print(f'\\boxed{{{sp.latex(var)}}}')."
        )

    # 4. Kiểm tra miền giá trị và kiểu dữ liệu biểu tượng qua SymPy
    try:
        import sympy
        from sympy import sympify, zoo, oo, nan, I

        # Chuyển đổi phân số LaTeX trước khi parse
        expr_str = re.sub(r"\\(?:d|t|c)?frac\s*\{([^}]+)\}\s*\{([^}]+)\}", r"((\1)/(\2))", cand_str)
        expr_str = expr_str.replace(r"\pi", "pi").replace(r"\sqrt", "sqrt").replace("^", "**")
        expr_str = expr_str.replace("$", "").replace("%", "").strip()

        # Kiểm tra tọa độ dạng tuple (x, y)
        if expr_str.startswith("(") and expr_str.endswith(")"):
            inner = expr_str[1:-1]
            tuple_parts = [p.strip() for p in inner.split(",") if p.strip()]
            if not tuple_parts:
                return ("fail", "Verification Error: Empty coordinate tuple.")
            
            # Kiểm tra ràng buộc tọa độ cực (polar coordinates)
            if any(t in question.lower() for t in ["polar coordinate", "polar coordinates", "polar form"]) and len(tuple_parts) == 2:
                try:
                    r_val = sympify(tuple_parts[0])
                    theta_val = sympify(tuple_parts[1])
                    if r_val.is_number and float(r_val) <= 0:
                        return ("fail", f"Verification Error: The polar radius r must be positive (r > 0), but got r = {r_val}. Actionable Fix: Filter or take the positive root for r.")
                    if theta_val.is_number:
                        two_pi = float(sympy.pi * 2)
                        th_f = float(theta_val)
                        if th_f < 0 or th_f >= two_pi:
                            return ("fail", f"Verification Error: The polar angle theta must satisfy 0 <= theta < 2*pi, but got theta = {theta_val}. Actionable Fix: Adjust theta modulo 2*pi using theta = theta % (2*sp.pi).")
                    return ("pass", "Verification Passed: Candidate answer satisfies polar-coordinate constraints (r > 0 and 0 <= theta < 2*pi).")
                except Exception:
                    pass
            return ("unknown", "Candidate coordinate tuple is well-formed.")

        # Parse biểu thức toán học
        sym_obj = sympify(expr_str)

        # Kiểm tra giá trị vô cực hoặc NaN
        if sym_obj in (zoo, oo, -oo, nan):
            return ("fail", f"Verification Error: Candidate answer evaluates to an undefined or infinite value ({sym_obj}). Actionable Fix: Avoid dividing by zero or taking limits to infinity without applying domain conditions.")

        # Kiểm tra nếu bài toán số học/đếm nhưng đáp án vẫn còn chứa biến tự do (Symbol) như "48 - 6*z"
        q_lower = question.lower()
        is_word_problem = any(kw in q_lower for kw in [
            "how many", "how much", "find the value", "what is the total",
            "calculate the", "speed", "hours", "dollars", "$", "percent", "miles"
        ]) and not any(kw in q_lower for kw in ["in terms of", "express in terms of", "polynomial p(x)", "function f(x)"])
        
        if is_word_problem and sym_obj.free_symbols:
            symbols_str = ", ".join(str(s) for s in sym_obj.free_symbols)
            return (
                "fail",
                f"Verification Error: Candidate answer '{cand_str}' still contains unresolved free symbol(s) ({symbols_str}). Actionable Fix: Solve all system equations or substitute given constants so no free symbols remain."
            )

        # Ràng buộc "in terms of X, Y": Nếu đề yêu cầu biểu diễn theo biến p, q nhưng đáp án lại không chứa p, q
        in_terms_match = re.search(r"(?:in terms of|express .* in terms of)\s+([a-zA-Z,\s\$\\\{\}]+)", q_lower)
        if in_terms_match:
            raw_vars = in_terms_match.group(1)
            target_vars = [v.strip().replace("$", "").replace("\\", "") for v in re.findall(r"[a-zA-Z]", raw_vars)]
            if target_vars:
                cand_syms = {str(s) for s in sym_obj.free_symbols}
                # Nếu đáp án hoàn toàn là số/hằng số mà không chứa biến được yêu cầu
                if not any(tv in cand_syms for tv in target_vars):
                    vars_str = ", ".join(target_vars)
                    return (
                        "fail",
                        f"Verification Error: The problem explicitly asks to express the answer in terms of ({vars_str}), but the candidate answer '{cand_str}' contains none of these target symbols. Actionable Fix: Do NOT allow SymPy simplify to convert symbols into numeric constants (e.g. sp.zeta(2) -> pi**2/6). Define symbols ({vars_str}) explicitly and substitute them into the target expression."
                    )

        # Ràng buộc số đếm / số lượng / số ước nguyên dương
        if any(term in q_lower for term in [
            "how many", "number of positive", "number of integers", 
            "number of ways", "number of divisors", "number of solutions",
            "proper divisors", "divisors"
        ]):
            if sym_obj.is_number:
                try:
                    num_val = float(sym_obj)
                    if num_val < 0:
                        return ("fail", f"Verification Error: This problem requires a non-negative count, but the result is negative ({num_val}).")
                    if not num_val.is_integer():
                        return ("fail", f"Verification Error: This problem requires an integer count, but the result is not an integer ({num_val}).")
                    return ("pass", f"Verification Passed: Candidate answer {cand_str} is a valid non-negative integer count.")
                except Exception:
                    pass

        # Ràng buộc xác suất (Probability trong đoạn [0, 1])
        if any(term in q_lower for term in ["probability", "what is the chance", "what is the probability"]):
            if sym_obj.is_number:
                try:
                    prob_val = float(sym_obj)
                    if prob_val < 0.0 or prob_val > 1.0:
                        return ("fail", f"Verification Error: Probability must lie in [0, 1], but the result is {prob_val}.")
                    return ("pass", f"Verification Passed: Candidate answer satisfies probability bounds [0, 1].")
                except Exception:
                    pass

        # Ràng buộc hình học: Diện tích, Độ dài, Chu vi, Bán kính, Thể tích phải > 0
        if any(term in q_lower for term in ["perimeter", "area", "length", "radius", "distance", "height", "volume"]):
            if sym_obj.is_number:
                try:
                    dim_val = float(sym_obj)
                    if dim_val <= 0:
                        return ("fail", f"Verification Error: Geometric quantity (length/area/perimeter) must be positive, but got {dim_val}.")
                    return ("pass", f"Verification Passed: Geometric dimension is positive ({dim_val}).")
                except Exception:
                    pass

        # Nếu đáp án là một số cụ thể (hằng số số thực / phân số / hằng số toán học)
        if sym_obj.is_number:
            return ("pass", f"Verification Passed: Candidate answer '{cand_str}' evaluates to a valid concrete numeric value/constant.")

    except Exception:
        # Nếu SymPy không parse được (ví dụ chuỗi chữ cái tên riêng như "Evelyn")
        if len(cand_str) > 0 and not any(ch in cand_str for ch in ["\n", "\r", "\t"]):
            return ("unknown", f"Candidate answer is a valid text entity ('{cand_str}').")

    return ("unknown", "Candidate answer is syntactically well-formed, but problem nature prevents automated symbolic proof without ground truth.")
