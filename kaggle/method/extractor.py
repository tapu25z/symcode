"""
Regex extraction utilities for boxed answers, python code blocks, and answer normalization.
Supports arithmetic, LaTeX fractions, coordinate tuples, and symbolic algebraic equivalence.
"""

import re
from typing import Optional, Any, Union, List, Dict


def extract_boxed_content(text: str) -> Optional[str]:
    """
    Extracts the content inside the last \\boxed{...} expression in the text.
    Handles nested braces properly using stack depth tracking.
    """
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        start = idx + len(r"\boxed{")
    elif text.rfind("\x08oxed{") != -1:
        idx = text.rfind("\x08oxed{")
        start = idx + len("\x08oxed{")
    elif text.rfind("boxed{") != -1:
        idx = text.rfind("boxed{")
        start = idx + len("boxed{")
    else:
        return None

    brace_depth = 1
    end = start
    while end < len(text) and brace_depth > 0:
        if text[end] == '{':
            brace_depth += 1
        elif text[end] == '}':
            brace_depth -= 1
        end += 1

    if brace_depth == 0:
        return text[start:end - 1].strip()
    return None


def extract_python_code(text: str) -> str:
    """
    Extracts Python code from markdown code blocks (```python ... ```).
    Falls back to raw code if no markdown block is found.
    """
    pattern = r"```(?:python|py)?\s*(.*?)\s*```"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    
    # Fallback: check if text has code-like lines
    lines = text.strip().split("\n")
    code_lines = [l for l in lines if not l.startswith("#") and any(k in l for k in ["import ", "def ", " = ", "print("])]
    if len(code_lines) >= 2:
        return text.strip()
    return text.strip()


def extract_gsm8k_ground_truth(answer_str: str) -> str:
    """
    Extracts numerical ground truth from GSM8K answer strings (after '####').
    """
    if "####" in answer_str:
        gt = answer_str.split("####")[-1].strip()
    else:
        gt = answer_str.strip()
    gt = gt.replace(",", "").replace("$", "").replace("%", "").strip()
    return gt


def extract_ground_truth(item_or_answer: Any) -> str:
    """
    Unified ground truth extractor for both GSM8K and MATH-500 datasets.
    """
    if isinstance(item_or_answer, dict):
        if "answer" in item_or_answer and item_or_answer["answer"]:
            ans = item_or_answer["answer"]
        elif "solution" in item_or_answer and item_or_answer["solution"]:
            ans = item_or_answer["solution"]
        else:
            ans = ""
    else:
        ans = str(item_or_answer)

    if "####" in ans:
        return extract_gsm8k_ground_truth(ans)
    
    boxed = extract_boxed_content(ans)
    if boxed is not None:
        return boxed.strip()

    return ans.strip()


def normalize_answer_str(ans: Optional[str]) -> str:
    """
    Normalizes prediction and ground truth strings for robust Exact Match (EM) comparison.
    Preserves commas for coordinate tuples while cleaning LaTeX wrappers.
    """
    if ans is None:
        return ""
    
    s = str(ans).strip()
    boxed = extract_boxed_content(s)
    if boxed is not None:
        s = boxed
        
    s = s.replace("$", "").replace("%", "").strip()
    # Remove leading/trailing formatting spaces
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = s.replace(r"\pi", "pi").replace(r"\degree", "deg")
    
    # Standardize LaTeX fractions \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    
    # Remove unnecessary spaces inside formulas
    s = re.sub(r"\s+", "", s)
    
    # Attempt float to int normalization if purely numeric (e.g. "42.0" -> "42")
    try:
        val = float(s)
        if val.is_integer():
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        pass

    return s


def check_exact_match(pred: Optional[str], gt: str) -> bool:
    """
    Checks whether the predicted answer matches the ground truth.
    Performs normalized string matching, coordinate tuple matching,
    and symbolic algebraic equivalence.
    """
    norm_pred = normalize_answer_str(pred)
    norm_gt = normalize_answer_str(gt)
    if not norm_pred or not norm_gt:
        return False
    
    # 1. Direct normalized match
    if norm_pred.lower() == norm_gt.lower():
        return True

    # 2. Substring / named entity matching (e.g. "Evelyn")
    if len(norm_gt) >= 3 and norm_gt.isalpha():
        if norm_gt.lower() in norm_pred.lower():
            return True

    # 3. Coordinate tuples comparison (e.g. (3, pi/2) vs (3, ((pi)/(2))))
    if norm_pred.startswith("(") and norm_pred.endswith(")") and norm_gt.startswith("(") and norm_gt.endswith(")"):
        pred_parts = [p.strip() for p in norm_pred[1:-1].split(",") if p.strip()]
        gt_parts = [p.strip() for p in norm_gt[1:-1].split(",") if p.strip()]
        if len(pred_parts) == len(gt_parts):
            if all(check_exact_match(p, g) for p, g in zip(pred_parts, gt_parts)):
                return True

    # 4. Symbolic algebraic equivalence via SymPy
    try:
        import sympy
        p_sym = sympy.sympify(norm_pred)
        g_sym = sympy.sympify(norm_gt)
        if sympy.simplify(p_sym - g_sym) == 0:
            return True
    except Exception:
        pass

    return False
