"""
Module trích xuất, chuẩn hóa đáp án và đánh giá Exact Match.
Hỗ trợ biểu thức số học, phân số LaTeX lồng nhau, tọa độ, tập hợp và tính tương đương đại số qua SymPy.
"""

import re
import ast
from typing import Optional, Any, Union, List, Dict
from fractions import Fraction


def extract_boxed_content(text: str) -> Optional[str]:
    """
    Trích xuất nội dung bên trong biểu thức \\boxed{...} cuối cùng trong văn bản.
    Xử lý chính xác các trường hợp khoảng trắng, dấu ngoặc lồng nhau và các biến thể LaTeX.
    """
    if not text or not isinstance(text, str):
        return None

    patterns = [
        r"\\boxed\s*\{",
        r"\x08oxed\s*\{",
        r"(?<![a-zA-Z0-9_\\])boxed\s*\{",
        r"\\fbox\s*\{",
        r"\\framebox\s*\{"
    ]
    
    matches = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            matches.append(m.end())
            
    if not matches:
        return None

    matches.sort()
    start = matches[-1]

    brace_depth = 1
    end = start
    while end < len(text) and brace_depth > 0:
        if text[end] == '{':
            brace_depth += 1
        elif text[end] == '}':
            brace_depth -= 1
        end += 1

    if brace_depth == 0:
        content = text[start:end - 1].strip()
        content = re.sub(r"^\\left\s*([(\[{|])", r"\1", content)
        content = re.sub(r"\\right\s*([)\]}|])$", r"\1", content)
        return content.strip()
        
    return None


def extract_answer_fallback(text: str) -> Optional[str]:
    """
    Trích xuất đáp án toàn diện cho CoT/Direct:
    1. Ưu tiên lấy nội dung trong \\boxed{...}.
    2. Nếu không có, tìm kiếm các mẫu kết luận chuẩn:
       'the answer is ...', 'the final answer is ...', '#### ...'.
    """
    if not text or not isinstance(text, str):
        return None
        
    text = remove_thinking_tags(text)
    boxed = extract_boxed_content(text)
    if boxed is not None:
        return boxed

    # Các mẫu kết luận phổ biến
    conclusion_patterns = [
        r"(?:the\s+final\s+answer\s+is|the\s+answer\s+is|is\s+equal\s+to|equals|is\s+therefore)\s*[:=]?\s*([^\n\r]+)",
        r"####\s*([^\n\r]+)"
    ]
    for pat in conclusion_patterns:
        matches = list(re.finditer(pat, text, flags=re.IGNORECASE))
        if matches:
            cand = matches[-1].group(1).strip()
            cand = cand.replace("$", "").rstrip(".").strip()
            # Lọc bỏ các từ thừa
            sub_m = re.search(r"(-?\d+(?:\.\d+)?(?:/\d+)?|[a-zA-Z]+)", cand)
            if sub_m:
                return sub_m.group(0).strip()
            return cand

    return None


def remove_thinking_tags(text: str) -> str:
    """Loại bỏ các thẻ <think>...</think> của các mô hình reasoning."""
    text = str(text or "")
    if "<think>" in text and "</think>" not in text:
        return text.split("<think>", 1)[0].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text.strip()


def extract_python_code(text: str) -> str:
    """
    Trích xuất mã nguồn Python từ khối markdown (```python ... ```).
    Ưu tiên lấy khối code chứa cú pháp thực thi hoặc SymPy.
    """
    if not text or not isinstance(text, str):
        return ""

    text = remove_thinking_tags(text)

    pattern = r"```(?:python|py)?\s*(.*?)\s*```"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        for m in reversed(matches):
            code_str = m.strip()
            if any(k in code_str for k in ["sympy", "sp.", "print(", "solve(", "symbols("]):
                return code_str
        return matches[-1].strip()
    
    lines = text.strip().split("\n")
    code_lines = [
        l for l in lines 
        if not l.startswith("#") and any(k in l for k in ["import ", "def ", " = ", "print(", "return ", "for ", "while "])
    ]
    if len(code_lines) >= 2:
        return text.strip()
        
    return text.strip()


def extract_symplanner_code(text: str) -> str:
    """
    Trích xuất và làm sạch mã nguồn Python dựa trên AST cho phương pháp SymPlanner (Divide-and-Plan).
    1. Trích xuất code thô bằng regex ```python ... ```.
    2. Thử ast.parse(candidate_code), nếu thành công trả về ngay.
    3. Nếu SyntaxError: Loại bỏ các dòng bắt đầu bằng Markdown headers (# Stage, ###, *),
       dòng tiêu đề hoặc văn bản rò rỉ từ Stage 1 (DIVIDE) / Stage 2 (PLAN).
    4. Thử ast.parse lại. Nếu vẫn lỗi, fallback về các dòng chứa từ khóa Python hợp lệ.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Trích xuất code thô bằng regex
    pattern = r"```(?:python|py)?\s*(.*?)\s*```"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        candidate_code = ""
        for m in reversed(matches):
            c_str = m.strip()
            if any(k in c_str for k in ["sympy", "sp.", "print(", "solve(", "symbols("]):
                candidate_code = c_str
                break
        if not candidate_code:
            candidate_code = matches[-1].strip()
    else:
        candidate_code = text.strip()

    if not candidate_code:
        return ""

    # 2. Thử ast.parse trực tiếp
    try:
        ast.parse(candidate_code)
        return candidate_code
    except SyntaxError:
        pass

    # 3. Làm sạch mạnh tay (Aggressive Sanitization)
    lines = candidate_code.split("\n")
    cleaned_lines = []
    
    # Regex nhận diện tiêu đề Markdown (# Stage, ###, *, Step) và công thức LaTeX (\[, \])
    md_header_or_stage_pattern = re.compile(
        r"^(?:\s*[*+-]\s+|#{1,6}\s*Stage|#{1,6}\s*Phase|#{1,6}\s+|Step\s*\d+\s*:|\\\[|\\\]|\$\$|\\\(|\\\))",
        re.IGNORECASE
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        
        # Loại bỏ các dòng bắt đầu bằng Markdown headers hoặc rò rỉ Stage 1/2
        if md_header_or_stage_pattern.match(stripped):
            continue
        if stripped.startswith(("\\[", "\\]", "$$", "\\(", "\\)")) or stripped.endswith(("\\]", "$$", "\\)")):
            continue
        if re.match(r"^(?:Stage\s*\d+|Phase\s*\d+|DIVIDE|PLAN|EXECUTE|Target\s*Unknown|Given\s*Quantities|Domain\s*Constraints)\s*:", stripped, re.IGNORECASE):
            continue
        if stripped in ["{", "}", "```", "```json", "```python"]:
            continue

        cleaned_lines.append(line)

    cleaned_code = "\n".join(cleaned_lines).strip()
    
    # 4. Thử ast.parse lại
    try:
        ast.parse(cleaned_code)
        return cleaned_code
    except SyntaxError:
        pass

    # 5. Fallback: Lọc các dòng chứa từ khóa / cấu trúc Python hợp lệ
    python_keywords = {
        "import ", "from ", "def ", "return ", "if ", "elif ", "else:",
        "for ", "while ", "try:", "except", "with ", "class ", "print(",
        "sp.", "sympy", " = ", "+=", "-=", "*=", "/=", "=="
    }
    fallback_lines = [
        l for l in cleaned_lines 
        if any(kw in l for kw in python_keywords) or l.strip().startswith("#")
    ]
    fallback_code = "\n".join(fallback_lines).strip()

    try:
        if fallback_code:
            ast.parse(fallback_code)
            return fallback_code
    except SyntaxError:
        pass

    return cleaned_code or candidate_code
def extract_gsm8k_ground_truth(answer_str: str) -> str:
    """
    Trích xuất đáp án chuẩn (ground truth) của GSM8K nằm sau ký hiệu '####'.
    """
    if not answer_str or not isinstance(answer_str, str):
        return ""

    if "####" in answer_str:
        gt = answer_str.split("####")[-1].strip()
    else:
        gt = answer_str.strip()
        
    gt = gt.replace(",", "").replace("$", "").replace("%", "").strip()
    return gt


def extract_ground_truth(item_or_answer: Any) -> str:
    """
    Bộ trích xuất đáp án chuẩn thống nhất cho GSM8K và MATH-500.
    """
    if isinstance(item_or_answer, dict):
        if item_or_answer.get("answer"):
            ans = str(item_or_answer["answer"])
        elif item_or_answer.get("solution"):
            ans = str(item_or_answer["solution"])
        else:
            ans = ""
    else:
        ans = str(item_or_answer) if item_or_answer is not None else ""

    if "####" in ans:
        return extract_gsm8k_ground_truth(ans)
    
    boxed = extract_boxed_content(ans)
    if boxed is not None:
        return boxed.strip()

    return ans.strip()


def _normalize_fractions(text: str) -> str:
    """
    Chuyển đổi đệ quy các phân số LaTeX (\\frac, \\dfrac, \\tfrac, \\cfrac) thành dạng (a)/(b).
    """
    # 1. First handle single-character fractions without braces (e.g. \frac43 -> ((4)/(3)))
    text = re.sub(r"\\(?:d|t|c)?frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])", r"((\1)/(\2))", text)
    
    # 2. Then handle standard brace fractions
    pattern = r"\\(?:d|t|c)?frac\s*\{((?:[^{}]|\{[^{}]*\})*)\}\s*\{((?:[^{}]|\{[^{}]*\})*)\}"
    prev = ""
    curr = text
    while prev != curr:
        prev = curr
        curr = re.sub(pattern, r"((\1)/(\2))", curr)
    return curr


def normalize_answer_str(ans: Optional[str]) -> str:
    """
    Chuẩn hóa chuỗi dự đoán và chuỗi ground truth để đánh giá Exact Match.
    Xử lý phân số, căn thức, góc độ, tọa độ, và loại bỏ wrapper LaTeX.
    """
    if ans is None:
        return ""

    if isinstance(ans, (list, tuple, set)):
        items = list(ans)
        left, right = ("(", ")") if isinstance(ans, tuple) else ("[", "]")
        return left + ",".join(normalize_answer_str(item) for item in items) + right
    
    s = str(ans).strip()
    boxed = extract_boxed_content(s)
    if boxed is not None:
        s = boxed

    if s.startswith(("[", "(")) and s.endswith(("]", ")")):
        try:
            literal = ast.literal_eval(s)
            if isinstance(literal, (list, tuple)) and not any(isinstance(item, (list, tuple, dict, set)) for item in literal):
                left, right = ("(", ")") if isinstance(literal, tuple) else ("[", "]")
                return left + ",".join(normalize_answer_str(item) for item in literal) + right
        except Exception:
            pass
        
    s = re.sub(r"^[a-zA-Z](?:\([a-zA-Z0-9_, ]+\))?\s*=\s*", "", s).strip()
    s = re.sub(r"^ans(?:wer)?\s*=\s*", "", s, flags=re.IGNORECASE).strip()

    s = s.replace("$", "").replace("%", "").strip()
    
    s = re.sub(r"\\(?:left|right|displaystyle|limits|textstyle|scriptstyle)", "", s)
    s = re.sub(r"\\(?:text|mathrm|mathbf|textbf|textit|operatorname|mbox)\s*\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|mathit)\s*i\b", "I", s, flags=re.IGNORECASE)
    s = re.sub(r"\\(?:quad|qquad|,|;|!|\s)", " ", s)
    
    s = re.sub(r"\^\s*\{\s*\\?(?:circ|degree)\s*\}", "", s)
    s = re.sub(r"\^\s*\\?(?:circ|degree)\b", "", s)
    s = re.sub(r"\\(?:degree|circ)\b", "", s)
    s = re.sub(r"\^deg\b", "", s)
    s = re.sub(r"\bdeg\b", "", s)
    s = re.sub(r"\\pi\b", "pi", s)
    
    s = s.replace(r"\{", "{").replace(r"\}", "}")
    
    s = _normalize_fractions(s)
    
    s = re.sub(r"\\sqrt\[([^\]]+)\]\{([^}]+)\}", r"((\2)**(1/(\1)))", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*([A-Za-z0-9]+)", r"sqrt(\1)", s)
    
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    # SymPy uses ``I`` for the imaginary unit; normalize common LaTeX/plain-text variants.
    s = re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "I", s)
    
    # Loại bỏ dấu phẩy ngăn cách hàng nghìn (ví dụ 1,200 -> 1200)
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)
    
    s = re.sub(r"\s+", "", s)
    
    # Loại bỏ đơn vị đo phổ biến ở cuối chuỗi nếu phía trước là số
    s = re.sub(r"(?<=\d)(?:hours?|hrs?|minutes?|mins?|seconds?|secs?|miles?|mph|meters?|cm|km|dollars?|cents?|eggs?|days?|weeks?|years?)$", "", s, flags=re.IGNORECASE)
    
    if s.endswith(".") and not re.search(r"\d\.\d", s):
        s = s[:-1].strip()

    try:
        val = float(s)
        if val.is_integer():
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        pass

    return s


def _split_top_level(text: str, separator: str = ",") -> List[str]:
    """Split a tuple/set/list without breaking nested function arguments."""
    parts: List[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == separator and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _parse_math_expression(text: str):
    """Parse a normalized expression with safe implicit multiplication support."""
    import sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
    local_dict = {
        "pi": sympy.pi,
        "sqrt": sympy.sqrt,
        "I": sympy.I,
        "E": sympy.E,
        "oo": sympy.oo,
    }
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    return parse_expr(text, local_dict=local_dict, transformations=transformations, evaluate=True)


def _collection_parts(text: str) -> Optional[List[str]]:
    if not text:
        return None
    if text[0] in "[{" and text[-1] in "]}":
        inner = text[1:-1]
        parts = _split_top_level(inner)
        return parts if parts else None
    if text[0] == "(" and text[-1] == ")":
        inner = text[1:-1]
        parts = _split_top_level(inner)
        return parts if len(parts) > 1 else None
    parts = _split_top_level(text)
    return parts if len(parts) > 1 else None


def check_exact_match(pred: Optional[str], gt: str) -> bool:
    """
    Kiểm tra độ chính xác tuyệt đối (Exact Match) giữa câu trả lời dự đoán và đáp án chuẩn.
    Thực hiện so sánh đa tầng: chuỗi chuẩn hóa, số học sai số e-6, tọa độ tuple, và đại số SymPy.
    """
    if pred is None or gt is None:
        return False
        
    norm_pred = normalize_answer_str(pred)
    norm_gt = normalize_answer_str(gt)
    
    if not norm_pred or not norm_gt:
        return False
    
    if norm_pred.lower() == norm_gt.lower():
        return True

    clean_p = re.sub(r"^[a-zA-Z]\s*=\s*", "", norm_pred).strip()
    clean_g = re.sub(r"^[a-zA-Z]\s*=\s*", "", norm_gt).strip()
    if clean_p.lower() == clean_g.lower():
        return True

    if len(norm_gt) >= 3 and norm_gt.isalpha():
        if norm_gt.lower() in norm_pred.lower():
            return True

    try:
        f_pred = float(norm_pred)
        f_gt = float(norm_gt)
        if abs(f_pred - f_gt) < 1e-5:
            return True
    except (ValueError, OverflowError):
        pass

    try:
        if "/" in norm_pred or "/" in norm_gt:
            fr_pred = Fraction(norm_pred.replace("(", "").replace(")", ""))
            fr_gt = Fraction(norm_gt.replace("(", "").replace(")", ""))
            if fr_pred == fr_gt:
                return True
    except Exception:
        pass

    try:
        import sympy
        pred_parts = _collection_parts(norm_pred)
        gt_parts = _collection_parts(norm_gt)
        if pred_parts or gt_parts:
            if pred_parts and not gt_parts and len(pred_parts) == 1:
                return check_exact_match(pred_parts[0], norm_gt)
            if gt_parts and not pred_parts and len(gt_parts) == 1:
                return check_exact_match(norm_pred, gt_parts[0])
            if pred_parts and gt_parts and len(pred_parts) == len(gt_parts) and all(check_exact_match(p, g) for p, g in zip(pred_parts, gt_parts)):
                return True
        p_sym = _parse_math_expression(norm_pred)
        g_sym = _parse_math_expression(norm_gt)
        diff = sympy.simplify(p_sym - g_sym)
        if diff == 0:
            return True
        if hasattr(diff, "evalf") and abs(float(diff.evalf())) < 1e-6:
            return True
    except Exception:
        pass

    return False
