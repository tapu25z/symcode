"""Unified GSM8K + Math500 answer equivalence scorer."""

from __future__ import annotations

import re
import math
from typing import Any, Callable

try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None


MATRIX_RE = re.compile(r"\\begin\{(?:p|b|v|V|B)?matrix\}(.*?)\\end\{(?:p|b|v|V|B)?matrix\}", re.S)
GSM8K_ANS_RE = re.compile(r"####\s*([^\n]+)")


def _clean_gsm8k_str(text: str) -> str:
    s = str(text or "").strip()
    match = GSM8K_ANS_RE.search(s)
    if match:
        s = match.group(1).strip()
    s = s.replace("$", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return s.strip()


def _sympify(text: Any):
    if sp is None or text in (None, ""):
        return None
    if isinstance(text, (list, tuple)):
        parsed_items = []
        for item in text:
            parsed = _sympify(item)
            if parsed is None:
                return None
            parsed_items.append(parsed)
        return sp.Tuple(*parsed_items)
    if isinstance(text, (sp.Basic, sp.MatrixBase, sp.Set)):
        return text
    s = _clean_gsm8k_str(str(text))
    s = re.sub(r"\\(?:left|right|displaystyle|limits|textstyle|scriptstyle)", "", s)
    s = re.sub(r"\\(?:text|mathrm|mathbf|textbf|textit|operatorname|mbox)\s*\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*(\d+)", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\infty", "oo").replace("∞", "oo").replace("π", "pi").replace("%", "")
    s = re.sub(r"(?<=[0-9)\]])\s*(?=(sqrt|sin|cos|tan|log|exp|pi|I)\b)", "*", s, flags=re.I)
    s = re.sub(r"(?<=[0-9)\]])\s*i\b", "*I", s, flags=re.I)
    s = re.sub(r"(?<=[0-9)\]])(?=I\b)", "*", s)
    s = re.sub(r"\bi\b", "I", s)
    locals_map = {
        "pi": sp.pi, "e": sp.E, "oo": sp.oo, "I": sp.I, "sqrt": sp.sqrt,
        "Matrix": sp.Matrix, "Tuple": sp.Tuple, "FiniteSet": sp.FiniteSet,
        "Interval": sp.Interval, "Union": sp.Union, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan
    }
    try:
        parsed = sp.sympify(s, locals=locals_map)
        if isinstance(parsed, tuple):
            parsed = sp.Tuple(*parsed)
        return parsed
    except Exception:
        return None


def _latex_matrix(text: str, normalize: Callable[[str], str]):
    if sp is None:
        return None
    match = MATRIX_RE.search(str(text))
    if not match:
        return None
    rows = []
    for raw_row in re.split(r"\\\\", match.group(1)):
        cells = [cell.strip() for cell in raw_row.split("&")]
        if cells and any(cells):
            row_items = []
            for cell in cells:
                c_clean = normalize(cell).replace(r"\infty", "oo").replace(" ", "")
                parsed_c = _sympify(c_clean) or _sympify(cell)
                row_items.append(parsed_c)
            rows.append(row_items)
    try:
        return sp.Matrix(rows) if rows else None
    except Exception:
        return None


def _is_structured(value: Any) -> bool:
    return isinstance(value, (tuple, list)) or (sp is not None and isinstance(value, (sp.Tuple, sp.MatrixBase, sp.Set)))


def _clean_base_suffix(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"_\{\s*\d+\s*\}", "", s)
    s = re.sub(r"_\d+\b", "", s)
    return s.strip()


def check_math500_equivalence(
    predicted: Any,
    canonical_answer: Any,
    ground_truth: Any,
    legacy_match: Callable[[Any, Any], bool],
    legacy_normalize: Callable[[Any], str],
) -> bool:
    if legacy_match(predicted, ground_truth):
        return True
    pred_text = str(predicted if predicted is not None else "")
    gold_text = str(ground_truth if ground_truth is not None else "")
    
    # GSM8K clean match
    clean_pred = _clean_gsm8k_str(pred_text)
    clean_gold = _clean_gsm8k_str(gold_text)
    if clean_pred and clean_gold and clean_pred == clean_gold:
        return True

    normalized_pred = legacy_normalize(pred_text).replace(r"\infty", "oo").replace("∞", "oo")
    normalized_gold = legacy_normalize(gold_text).replace(r"\infty", "oo").replace("∞", "oo")
    if normalized_pred == normalized_gold:
        return True
    if _clean_base_suffix(normalized_pred) == _clean_base_suffix(normalized_gold) and _clean_base_suffix(normalized_pred):
        return True

    # Check matrix equivalence
    try:
        pred_matrix = _sympify(str(canonical_answer)) if "Matrix(" in str(canonical_answer) else (_sympify(pred_text) if "Matrix(" in pred_text else _latex_matrix(pred_text, legacy_normalize))
        gold_matrix = _latex_matrix(gold_text, legacy_normalize) or _sympify(gold_text)
        if isinstance(pred_matrix, sp.MatrixBase) and isinstance(gold_matrix, sp.MatrixBase) and pred_matrix.shape == gold_matrix.shape:
            if pred_matrix == gold_matrix or all(sp.simplify(a - b) == 0 for a, b in zip(list(pred_matrix), list(gold_matrix))):
                return True
    except Exception:
        pass

    # Check candidate values against gold
    candidates = [canonical_answer, predicted, clean_pred]
    gold_expr = _sympify(normalized_gold) or _sympify(gold_text) or _sympify(clean_gold)

    for cand in candidates:
        if cand in (None, ""):
            continue
        cand_expr = _sympify(cand)
        if cand_expr is None:
            cand_str = str(cand)
            cand_expr = _sympify(cand_str) or _sympify(legacy_normalize(cand_str))
        if cand_expr is not None and gold_expr is not None:
            if cand_expr == gold_expr:
                return True
            if isinstance(cand_expr, (tuple, list, sp.Tuple)) and isinstance(gold_expr, (tuple, list, sp.Tuple)) and len(cand_expr) == len(gold_expr):
                elem_matches = []
                for c_elem, g_elem in zip(cand_expr, gold_expr):
                    try:
                        if c_elem == g_elem or sp.simplify(c_elem - g_elem) == 0:
                            elem_matches.append(True)
                        elif abs(complex(sp.N(c_elem - g_elem))) <= 1e-6:
                            elem_matches.append(True)
                        else:
                            elem_matches.append(False)
                    except Exception:
                        elem_matches.append(False)
                if all(elem_matches):
                    return True
            # Never subtract a structured SymPy object from a scalar. Older
            # code let Tuple reach this branch, which emits a deprecation
            # warning and can become a hard error in newer SymPy releases.
            if not _is_structured(cand_expr) and not _is_structured(gold_expr):
                try:
                    diff = sp.simplify(cand_expr - gold_expr)
                    if diff == 0 or getattr(diff, "equals", lambda _: False)(0):
                        return True
                except Exception:
                    pass
                try:
                    diff_val = complex(sp.N(cand_expr - gold_expr))
                    if abs(diff_val) <= 1e-6:
                        return True
                except Exception:
                    pass

    return False
