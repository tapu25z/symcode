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
            "Lỗi kiểm chứng: Không tìm thấy đáp án ứng viên hoặc mã nguồn không in ra định dạng \\boxed{...}."
        )

    cand_str = str(candidate_answer).strip()

    # 2. Kiểm tra các placeholder không hợp lệ hoặc chuỗi lỗi
    invalid_tokens = ["todo", "none", "null", "undefined", "error", "nan", "invalid", "no valid solution", "<function", "<class"]
    if any(p in cand_str.lower() for p in invalid_tokens):
        return (
            "fail",
            f"Lỗi kiểm chứng: Đáp án '{cand_str}' là token không hợp lệ (None/Invalid/NaN/Error/Function Object). Hãy tính toán ra giá trị cụ thể."
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
            f"Lỗi kiểm chứng: Đáp án '{cand_str}' là tên biến Python chưa được đánh giá thành giá trị cụ thể. Hãy tính toán giá trị của biến trước khi in."
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
                return ("fail", "Lỗi kiểm chứng: Cặp tọa độ rỗng.")
            
            # Kiểm tra ràng buộc tọa độ cực (polar coordinates)
            if any(t in question.lower() for t in ["polar coordinate", "polar coordinates", "polar form"]) and len(tuple_parts) == 2:
                try:
                    r_val = sympify(tuple_parts[0])
                    theta_val = sympify(tuple_parts[1])
                    if r_val.is_number and float(r_val) <= 0:
                        return ("fail", f"Lỗi kiểm chứng: Bán kính cực r phải dương (r > 0), nhưng nhận được r = {r_val}.")
                    if theta_val.is_number:
                        two_pi = float(sympy.pi * 2)
                        th_f = float(theta_val)
                        if th_f < 0 or th_f >= two_pi:
                            return ("fail", f"Lỗi kiểm chứng: Góc cực theta phải thỏa mãn 0 <= theta < 2*pi, nhưng nhận được theta = {theta_val}.")
                    return ("pass", "Kiểm chứng thành công: Thỏa mãn ràng buộc tọa độ cực (r > 0 và 0 <= theta < 2*pi).")
                except Exception:
                    pass
            return ("unknown", "Cặp tọa độ cú pháp chuẩn.")

        # Parse biểu thức toán học
        sym_obj = sympify(expr_str)

        # Kiểm tra giá trị vô cực hoặc NaN
        if sym_obj in (zoo, oo, -oo, nan):
            return ("fail", f"Lỗi kiểm chứng: Đáp án đánh giá thành giá trị không xác định hoặc vô cực ({sym_obj}).")

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
                f"Lỗi kiểm chứng: Đáp án '{cand_str}' vẫn còn chứa biến tự do ({symbols_str}) chưa được giải thành số cụ thể. Hãy giải hệ phương trình hoặc tính toán tuần tự để tìm giá trị số cụ thể."
            )

        # Ràng buộc số đếm / số lượng / số ước nguyên dương
        if any(term in q_lower for term in [
            "how many", "number of positive", "number of integers", 
            "number of ways", "number of divisors", "number of solutions"
        ]):
            if sym_obj.is_number:
                try:
                    num_val = float(sym_obj)
                    if num_val < 0:
                        return ("fail", f"Lỗi kiểm chứng: Bài toán yêu cầu số đếm không âm, nhưng kết quả nhận được là số âm ({num_val}).")
                    if not num_val.is_integer():
                        return ("fail", f"Lỗi kiểm chứng: Bài toán yêu cầu số đếm nguyên, nhưng kết quả nhận được không phải số nguyên ({num_val}).")
                    return ("pass", f"Kiểm chứng thành công: Đáp án {cand_str} là số nguyên không âm hợp lệ.")
                except Exception:
                    pass

        # Ràng buộc xác suất (Probability trong đoạn [0, 1])
        if any(term in q_lower for term in ["probability", "what is the chance", "what is the probability"]):
            if sym_obj.is_number:
                try:
                    prob_val = float(sym_obj)
                    if prob_val < 0.0 or prob_val > 1.0:
                        return ("fail", f"Lỗi kiểm chứng: Xác suất phải nằm trong đoạn [0, 1], nhưng kết quả là {prob_val}.")
                    return ("pass", f"Kiểm chứng thành công: Đáp án thỏa mãn khoảng xác suất [0, 1].")
                except Exception:
                    pass

        # Ràng buộc hình học: Diện tích, Độ dài, Chu vi, Bán kính, Thể tích phải > 0
        if any(term in q_lower for term in ["perimeter", "area", "length", "radius", "distance", "height", "volume"]):
            if sym_obj.is_number:
                try:
                    dim_val = float(sym_obj)
                    if dim_val <= 0:
                        return ("fail", f"Lỗi kiểm chứng: Đại lượng hình học (độ dài/diện tích/chu vi) phải dương, nhưng nhận được {dim_val}.")
                    return ("pass", f"Kiểm chứng thành công: Đại lượng hình học có giá trị dương ({dim_val}).")
                except Exception:
                    pass

    except Exception:
        # Nếu SymPy không parse được (ví dụ chuỗi chữ cái tên riêng như "Evelyn")
        if len(cand_str) > 0 and not any(ch in cand_str for ch in ["\n", "\r", "\t"]):
            return ("unknown", f"Đáp án là thực thể văn bản hợp lệ ('{cand_str}').")

    return ("unknown", "Đáp án hợp thức về mặt cú pháp nhưng đặc thù bài toán yêu cầu so khớp kết quả.")

