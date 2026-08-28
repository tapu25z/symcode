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
    if sp is None:
        return None
    locals_map = {"pi": sp.pi, "e": sp.E, "oo": sp.oo, "sqrt": sp.sqrt, "Matrix": sp.Matrix, "Tuple": sp.Tuple, "FiniteSet": sp.FiniteSet, "Interval": sp.Interval, "Union": sp.Union}
    return sp.sympify(text, locals=locals_map)


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


def check_math500_equivalence(
    predicted: Any,
    canonical_answer: Any,
    ground_truth: Any,
    legacy_match: Callable[[Any, Any], bool],
    legacy_normalize: Callable[[Any], str],
) -> bool:
    if legacy_match(predicted, ground_truth):
        return True
    pred_text, gold_text = str(predicted or ""), str(ground_truth or "")
    normalized_pred = legacy_normalize(pred_text).replace(r"\infty", "oo").replace("∞", "oo")
    normalized_gold = legacy_normalize(gold_text).replace(r"\infty", "oo").replace("∞", "oo")
    if normalized_pred == normalized_gold:
        return True
    try:
        pred_matrix = _sympify(str(canonical_answer)) if "Matrix(" in str(canonical_answer) else _latex_matrix(pred_text, legacy_normalize)
        gold_matrix = _latex_matrix(gold_text, legacy_normalize)
        if pred_matrix is not None and gold_matrix is not None and pred_matrix == gold_matrix:
            return True
    except Exception:
        pass
    try:
        pred_expr = _sympify(str(canonical_answer) if canonical_answer not in (None, "") else normalized_pred)
        gold_expr = _sympify(normalized_gold)
        if pred_expr == gold_expr:
            return True
        return bool(sp.simplify(pred_expr - gold_expr) == 0)
    except Exception:
        return False
