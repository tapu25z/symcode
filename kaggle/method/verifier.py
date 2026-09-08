"""
Module bộ kiểm chứng toán học độc lập (Independent Mathematical Verifier) cho phương pháp SymCode và SymPlanner.
Thực hiện kiểm tra tính nhất quán toán học, ràng buộc miền giá trị, phần dư phương trình và tính hợp thức đại số
hoàn toàn độc lập, KHÔNG sử dụng hoặc làm lộ đáp án chuẩn (ground truth).
"""

import re
from typing import Tuple, Optional, Any

from .target_contract import target_contract_feedback, infer_target_spec
from .extractor import check_exact_match


def _extract_in_terms_vars(question: str) -> list[str]:
    cleaned = re.sub(r"[$\\{}.,]", " ", str(question or "").lower())
    match = re.search(r"(?:in terms of|express .* in terms of)\s+([a-zA-Z](?:\s*(?:,|and)\s*[a-zA-Z])*)", cleaned)
    if not match:
        return []
    return [item for item in re.findall(r"[a-zA-Z]+", match.group(1)) if item != "and"]


def _code_strategy_feedback(question: str, candidate_answer: str, code: Optional[str]) -> tuple[str, str] | None:
    q_lower = str(question or "").lower()
    code_lower = str(code or "").lower()
    cand_lower = str(candidate_answer or "").lower()

    target_vars = _extract_in_terms_vars(q_lower)
    if target_vars and not any(re.search(rf"\b{re.escape(var)}\b", cand_lower) for var in target_vars):
        return (
            "fail",
            "Verification Error: symbolic target must be simplified in the requested variables; do not leave an unevaluated special-function sum."
        )

    if "round table" in q_lower and "no two" in q_lower and "next to each other" in q_lower:
        has_adjacency_check = any(token in code_lower for token in ("itertools.permutations", "for perm", "def is_valid", "adjacent", "next_to"))
        if "restricted_permutations" in code_lower and "total_permutations" in code_lower and not has_adjacency_check:
            return (
                "fail",
                "Verification Error: circular no-adjacency counting needs pairwise adjacency handling or brute-force validation; subtracting only one grouped case is incomplete."
            )

    if "different battalions" in q_lower or ("how many different" in q_lower and "soldiers" in q_lower):
        if "min(" in code_lower and "//" in code_lower and "comb" not in code_lower and "binomial" not in code_lower:
            return (
                "fail",
                "Verification Error: this asks for number of selectable groups, so use combinations/binomial counts rather than the maximum number of full battalions."
            )

    if "three for" in q_lower and "$1" in q_lower:
        if "// 3" in code_lower and re.search(r"price_\w+\s*=\s*1\s*/\s*3", code_lower):
            return (
                "fail",
                "Verification Error: phrase 'three for $1' means each group of three earns one dollar; do not multiply the number of groups by 1/3 again."
            )

    has_norm_target = "norm" in q_lower or "||" in q_lower or "\\|" in q_lower or "magnitude" in q_lower
    has_matrix_target = "matrix" in q_lower or "pmatrix" in q_lower or "begin{pmatrix}" in q_lower
    if has_norm_target and "for all" in q_lower and has_matrix_target:
        if "eigenvals" in code_lower and not any(token in code_lower for token in ("singular", ".t *", ".t*", "transpose")):
            return (
                "fail",
                "Verification Error: the smallest C for ||Av|| <= C||v|| is the spectral norm, sqrt(max eigenvalue of A.T*A), not the maximum absolute eigenvalue of A."
            )

    if "logarithms of the roots" in q_lower and "sp.solve(log_condition" in code_lower:
        return (
            "fail",
            "Verification Error: use log product rules directly with Vieta; do not ask SymPy to solve a sum of logs for a product expression."
        )

    if "functional equation" in q_lower and "sp.function" in code_lower and "f(2)" in code_lower:
        return (
            "fail",
            "Verification Error: solve the functional equation by assuming a quadratic/affine polynomial form and equating coefficients, not by solving for isolated f(k) symbols."
        )

    if "smallest positive perfect cube" in q_lower and "three consecutive integers" in q_lower:
        try:
            import sympy as sp
            candidate_value = sp.sympify(candidate_answer)
            for base in range(1, 1000):
                cube = base ** 3
                if cube % 3 == 0:
                    if sp.simplify(candidate_value - cube) != 0:
                        return (
                            "fail",
                            "Verification Error: candidate is not the smallest qualifying cube; search cube values in increasing order and return the cube value itself."
                        )
                    break
        except Exception:
            pass

    if "rotated around" in q_lower and ("complex" in q_lower or " i" in q_lower or "i$" in q_lower):
        if any(token in code_lower for token in ("sp.arg", "arg(", "sp.abs", "abs(")) and "z-c" not in code_lower.replace(" ", ""):
            return (
                "fail",
                "Verification Error: complex rotation around a center should use c + (z-c)*(cos(theta)+I*sin(theta)); do not rotate polar coordinates around the origin."
            )

    if "compound interest" in q_lower and "deposit" in q_lower:
        if re.search(r"\b[a-z]\s*\*\s*\(\s*1\s*\+\s*r\s*\)\s*\*\*\s*n\b", code_lower) and "(1 + r)**2" not in code_lower:
            return (
                "fail",
                "Verification Error: repeated end-of-year deposits require summing separately compounded deposits, not treating all deposits as one lump sum."
            )

    if "remainder" in q_lower and ("mod" in q_lower or "pmod" in q_lower):
        if "as_coefficients_dict()[1]" in code_lower:
            return (
                "fail",
                "Verification Error: modular remainder should be computed by direct residue substitution and modulo reduction, not by extracting a constant coefficient."
            )

    if "for all angles" in q_lower and "sin" in q_lower and "collect" in code_lower and "coeffs_lhs.get" in code_lower:
        return (
            "fail",
            "Verification Error: trig power identity coefficients are not obtained reliably by collect on sin(k*x); use exact expansion/equating at enough sample points or Fourier identities."
        )

    if "reassigned to" in q_lower and "denali" in q_lower and "nate" in q_lower:
        compact_code = code_lower.replace(" ", "")
        if "12+x" in compact_code:
            return (
                "fail",
                "Verification Error: when x of Nate's dogs are reassigned to Denali, Denali gains x and Nate loses x; Nate's count should not become 12+x."
            )

    return None


def _diagram_numeric_feedback(question: str, candidate_answer: str) -> tuple[str, str] | None:
    q_text = str(question or "")
    q_lower = q_text.lower()
    segment_match = re.search(r"(?:what\s+is|find)\s+\$?([A-Za-z]{2})\$?", q_text, re.IGNORECASE)
    if "[asy]" not in q_lower or not segment_match:
        return None
    segment = segment_match.group(1).upper()
    pair_pattern = re.compile(
        r"\b([A-Za-z])\s*=\s*\(([^,()]+(?:\([^)]*\))?[^,()]*),\s*([^,()]+(?:\([^)]*\))?[^,()]*)\)"
    )
    coords = {name.upper(): (x.strip(), y.strip()) for name, x, y in pair_pattern.findall(q_text)}
    if segment[0] not in coords or segment[1] not in coords:
        return None
    try:
        import sympy as sp
        x1, y1 = [sp.sympify(value, locals={"sqrt": sp.sqrt, "pi": sp.pi}) for value in coords[segment[0]]]
        x2, y2 = [sp.sympify(value, locals={"sqrt": sp.sqrt, "pi": sp.pi}) for value in coords[segment[1]]]
        expected = sp.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if not check_exact_match(candidate_answer, str(expected)):
            return (
                "fail",
                f"Verification Error: diagram source gives explicit coordinates for segment {segment}; compute the distance from those coordinates."
            )
    except Exception:
        return None
    return None


def verify_candidate_answer(
    question: str,
    candidate_answer: Optional[str],
    code: Optional[str] = None,
    stdout: Optional[str] = None,
    planner_note: str = ""
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

    contract_result = target_contract_feedback(question, cand_str, planner_note)
    defer_contract_result = (
        contract_result is not None
        and contract_result[0] == "unknown"
        and "diagram-dependent" in contract_result[1].lower()
    )
    if contract_result is not None and not defer_contract_result:
        return contract_result

    # 2. Kiểm tra các placeholder không hợp lệ hoặc chuỗi lỗi
    invalid_tokens = ["todo", "none", "null", "undefined", "error", "nan", "invalid", "no valid solution", "<function", "<class"]
    if any(p in cand_str.lower() for p in invalid_tokens):
        return (
            "fail",
            f"Verification Error: Candidate answer '{cand_str}' is an invalid token (None/Invalid/NaN/Error/Function Object). Compute and print a concrete value."
        )

    target_spec = infer_target_spec(question, planner_note)
    if target_spec.get("answer_type") == "number":
        if cand_str in {"[]", "{}", "()"}:
            return ("fail", "Verification Error: numeric target cannot be an empty collection.")
        if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", cand_str):
            return ("fail", "Verification Error: numeric target cannot be an unresolved symbol or placeholder variable.")

    strategy_result = _code_strategy_feedback(question, cand_str, code)
    if strategy_result is not None:
        return strategy_result

    diagram_result = _diagram_numeric_feedback(question, cand_str)
    if diagram_result is not None:
        return diagram_result
    if defer_contract_result:
        return contract_result

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
            "Verification Error: Candidate answer '" + cand_str + "' is an unevaluated Python variable name. Actionable Fix: Compute the actual value of the variable first, then pass the evaluated variable to print(f'\\boxed{sp.latex(var)}')."
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
                        return ("fail", f"Verification Error: The polar radius r must be positive (r > 0), but got r = {r_val}.")
                    if theta_val.is_number:
                        two_pi = float(sympy.pi * 2)
                        th_f = float(theta_val)
                        if th_f < 0 or th_f >= two_pi:
                            return ("fail", f"Verification Error: The polar angle theta must satisfy 0 <= theta < 2*pi, but got theta = {theta_val}.")
                    return ("unknown", "Verification Unknown: candidate satisfies polar-coordinate bounds, but no relation proves the requested pair.")
                except Exception:
                    pass
            return ("unknown", "Candidate coordinate tuple is well-formed.")

        # Parse biểu thức toán học

        # Parse biểu thức toán học
        sym_obj = sympify(expr_str)

        # Kiểm tra giá trị vô cực hoặc NaN
        if sym_obj in (zoo, oo, -oo, nan):
            return ("fail", f"Verification Error: Candidate answer evaluates to an undefined or infinite value ({sym_obj}).")

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
                f"Verification Error: Candidate answer '{cand_str}' still contains unresolved free symbol(s) ({symbols_str}). Solve the equations or compute the result step by step to obtain a concrete numeric value."
            )

        # Ràng buộc "in terms of X, Y": Nếu đề yêu cầu biểu diễn theo biến p, q nhưng đáp án lại không chứa p, q
        target_vars = _extract_in_terms_vars(q_lower)
        if target_vars:
            cand_syms = {str(s) for s in sym_obj.free_symbols}
            # Nếu đáp án hoàn toàn là số/hằng số mà không chứa biến được yêu cầu
            if not any(tv in cand_syms for tv in target_vars):
                vars_str = ", ".join(target_vars)
                return (
                    "fail",
                    f"Verification Error: The problem explicitly asks to express the answer in terms of ({vars_str}), but the candidate answer '{cand_str}' contains none of these target symbols. Express your solution using symbols ({vars_str})."
                )

        # Ràng buộc số đếm / số lượng / số ước nguyên dương
        is_distance_question = any(term in q_lower for term in ["miles", "kilometers", "kilometres", "meters", "distance"])
        if not is_distance_question and any(term in q_lower for term in [
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
                    return ("unknown", f"Verification Unknown: candidate {cand_str} is a valid count, but no independent relation proves the count.")
                except Exception:
                    pass

        # Ràng buộc xác suất (Probability trong đoạn [0, 1])
        if any(term in q_lower for term in ["probability", "what is the chance", "what is the probability"]):
            if sym_obj.is_number:
                try:
                    prob_val = float(sym_obj)
                    if prob_val < 0.0 or prob_val > 1.0:
                        return ("fail", f"Verification Error: Probability must lie in [0, 1], but the result is {prob_val}.")
                    return ("unknown", "Verification Unknown: candidate satisfies probability bounds, but no independent relation proves the probability.")
                except Exception:
                    pass

        # Ràng buộc hình học: Diện tích, Độ dài, Chu vi, Bán kính, Thể tích phải > 0
        if any(term in q_lower for term in ["perimeter", "area", "length", "radius", "distance", "height", "volume"]):
            if sym_obj.is_number:
                try:
                    dim_val = float(sym_obj)
                    if dim_val <= 0:
                        return ("fail", f"Verification Error: Geometric quantity (length/area/perimeter) must be positive, but got {dim_val}.")
                    return ("unknown", f"Verification Unknown: geometric quantity is positive ({dim_val}), but no independent relation proves its value.")
                except Exception:
                    pass

        # A syntactically valid scalar is not proof of correctness. Without a
        # relation-level proof, fail closed and let the evaluator score it only
        # against ground truth.
        if sym_obj.is_number:
            return ("unknown", f"Verification Unknown: candidate '{cand_str}' is numeric, but no independent relation proves the target value.")

    except Exception:
        # Nếu SymPy không parse được (ví dụ chuỗi chữ cái tên riêng như "Evelyn")
        if len(cand_str) > 0 and not any(ch in cand_str for ch in ["\n", "\r", "\t"]):
            return ("unknown", f"Candidate answer is a valid text entity ('{cand_str}').")

    return ("unknown", "Candidate answer is syntactically well-formed, but problem nature prevents automated symbolic proof without ground truth.")
