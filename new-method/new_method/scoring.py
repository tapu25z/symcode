"""Math500-aware answer equivalence layered on top of the legacy scorer."""

from __future__ import annotations

import re
from typing import Any, Callable

try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None


MATRIX_RE = re.compile(r"\\begin\{(?:p|b|v|V|B)?matrix\}(.*?)\\end\{(?:p|b|v|V|B)?matrix\}", re.S)


def _sympify(text: str):
    if sp is None or not text:
        return None
    s = str(text).strip()
    s = re.sub(r"\\(?:left|right|displaystyle|limits|textstyle|scriptstyle)", "", s)
    s = re.sub(r"\\(?:text|mathrm|mathbf|textbf|textit|operatorname|mbox)\s*\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*(\d+)", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\infty", "oo").replace("∞", "oo").replace("π", "pi").replace("$", "").replace("%", "")
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
        return sp.sympify(s, locals=locals_map)
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
            rows.append([_sympify(normalize(cell).replace(r"\infty", "oo")) for cell in cells])
    return sp.Matrix(rows) if rows else None


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
    pred_text, gold_text = str(predicted if predicted is not None else ""), str(ground_truth if ground_truth is not None else "")
    normalized_pred = legacy_normalize(pred_text).replace(r"\infty", "oo").replace("∞", "oo")
    normalized_gold = legacy_normalize(gold_text).replace(r"\infty", "oo").replace("∞", "oo")
    if normalized_pred == normalized_gold:
        return True
    if _clean_base_suffix(normalized_pred) == _clean_base_suffix(normalized_gold) and _clean_base_suffix(normalized_pred):
        return True

    # Check matrix equivalence
    try:
        pred_matrix = _sympify(str(canonical_answer)) if "Matrix(" in str(canonical_answer) else _latex_matrix(pred_text, legacy_normalize)
        gold_matrix = _latex_matrix(gold_text, legacy_normalize)
        if pred_matrix is not None and gold_matrix is not None and pred_matrix == gold_matrix:
            return True
    except Exception:
        pass

    # Check candidate values against gold
    candidates = [canonical_answer, predicted]
    gold_expr = _sympify(normalized_gold) or _sympify(gold_text)

    for cand in candidates:
        if cand in (None, ""):
            continue
        cand_str = str(cand)
        cand_expr = _sympify(cand_str) or _sympify(legacy_normalize(cand_str))
        if cand_expr is not None and gold_expr is not None:
            if cand_expr == gold_expr:
                return True
            try:
                diff = sp.simplify(cand_expr - gold_expr)
                if diff == 0 or getattr(diff, "equals", lambda _: False)(0):
                    return True
                if hasattr(diff, "evalf"):
                    val = diff.evalf()
                    if val.is_number and abs(complex(val)) < 1e-4:
                        return True
            except Exception:
                pass
            try:
                c_val = complex(cand_expr.evalf())
                g_val = complex(gold_expr.evalf())
                if abs(c_val - g_val) < 1e-4:
                    return True
            except Exception:
                pass

    return False

