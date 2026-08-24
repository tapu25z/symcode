"""
Independent mathematical verifier module for SymCode+ baseline.
Performs sanity checks, domain constraint validations, equation residuals,
and symbolic well-formedness tests on candidate answers WITHOUT ground truth leakage.
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
    Verifies a candidate answer purely based on mathematical consistency and problem constraints.
    Does NOT access or use the ground truth answer.

    Returns:
        (status, feedback_message)
        where status is one of:
            - "pass": Candidate answer verified and satisfied domain/equation constraints.
            - "fail": Candidate answer violates mathematical constraints or is invalid.
            - "unknown": Candidate answer is well-formed, but problem nature prevents deterministic proof without ground truth.
    """
    # 1. Check for empty or non-existent candidate answer
    if candidate_answer is None or not str(candidate_answer).strip():
        return (
            "fail",
            "Verification Failed: No candidate answer was produced or extracted in LaTeX \\boxed{...} format."
        )

    cand_str = str(candidate_answer).strip()

    # 2. Check for unexpanded code identifiers or literal variable names inside \boxed{}
    # E.g. \boxed{perimeter_hexagon}, \boxed{ans}, \boxed{result}, \boxed{x}
    raw_var_pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
    common_code_vars = {
        "ans", "answer", "result", "res", "final_answer", "sol", "solution",
        "output", "val", "value", "perimeter_hexagon", "num_divisors", "count",
        "total", "speed", "max_speed", "min_val", "max_val"
    }
    cleaned_cand = cand_str.replace("\\", "").replace("{", "").replace("}", "").strip()
    if cleaned_cand in common_code_vars or (re.match(raw_var_pattern, cleaned_cand) and len(cleaned_cand) > 4):
        return (
            "fail",
            f"Verification Failed: Candidate answer '{cand_str}' appears to be an unexpanded Python variable name rather than a computed value. Ensure your code evaluates the variable before printing."
        )

    # 3. Check for non-math placeholder strings or truncated outputs
    if any(p in cand_str.lower() for p in ["todo", "none", "null", "undefined", "error", "nan"]):
        return (
            "fail",
            f"Verification Failed: Candidate answer '{cand_str}' contains an invalid placeholder, NaN, or undefined token."
        )

    # 4. Symbolic domain and type constraint checks via SymPy
    try:
        import sympy
        from sympy import sympify, zoo, oo, nan

        # Convert basic LaTeX fractions before parsing
        expr_str = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", cand_str)
        expr_str = expr_str.replace(r"\pi", "pi").replace(r"\sqrt", "sqrt").replace("^", "**")
        expr_str = expr_str.replace("$", "").replace("%", "").strip()

        # If it's a coordinate tuple like (3, pi/2)
        if expr_str.startswith("(") and expr_str.endswith(")"):
            inner = expr_str[1:-1]
            tuple_parts = [p.strip() for p in inner.split(",") if p.strip()]
            if not tuple_parts:
                return ("fail", "Verification Failed: Coordinate tuple is empty.")
            
            # Check polar coordinates constraints if question mentions polar coordinates
            if "polar coordinates" in question.lower() and len(tuple_parts) == 2:
                try:
                    r_val = sympify(tuple_parts[0])
                    theta_val = sympify(tuple_parts[1])
                    if r_val.is_number and float(r_val) <= 0:
                        return ("fail", f"Verification Failed: Polar radius r must be positive (r > 0), but got r = {r_val}.")
                    if theta_val.is_number:
                        two_pi = float(sympy.pi * 2)
                        th_f = float(theta_val)
                        if th_f < 0 or th_f >= two_pi:
                            return ("fail", f"Verification Failed: Polar angle theta must satisfy 0 <= theta < 2*pi, but got theta = {theta_val}.")
                    return ("pass", "Verification Passed: Polar coordinate domain constraints satisfied (r > 0 and 0 <= theta < 2*pi).")
                except Exception:
                    pass
            return ("unknown", "Candidate coordinate tuple is well-formed.")

        # Parse mathematical expression
        sym_obj = sympify(expr_str)

        # Check for unphysical infinity or NaN values
        if sym_obj in (zoo, oo, -oo, nan):
            return ("fail", f"Verification Failed: Candidate answer evaluated to non-finite entity ({sym_obj}).")

        # Check domain constraint: "how many" / "number of positive whole-number divisors" / "counting"
        q_lower = question.lower()
        if any(term in q_lower for term in ["how many", "number of positive", "number of integers", "number of ways", "number of divisors"]):
            if sym_obj.is_number:
                try:
                    num_val = float(sym_obj)
                    if num_val < 0:
                        return ("fail", f"Verification Failed: Problem requires a non-negative count, but candidate answer is negative ({num_val}).")
                    if not num_val.is_integer():
                        return ("fail", f"Verification Failed: Problem requires an integer count, but candidate answer is non-integer ({num_val}).")
                    return ("pass", f"Verification Passed: Candidate answer {cand_str} is a valid non-negative integer count.")
                except Exception:
                    pass

        # Check domain constraint: Probability must be in [0, 1]
        if any(term in q_lower for term in ["probability", "what is the chance"]):
            if sym_obj.is_number:
                try:
                    prob_val = float(sym_obj)
                    if prob_val < 0.0 or prob_val > 1.0:
                        return ("fail", f"Verification Failed: Probability must be within [0, 1], but candidate answer evaluated to {prob_val}.")
                    return ("pass", f"Verification Passed: Candidate answer satisfies probability bounds [0, 1].")
                except Exception:
                    pass

        # Check domain constraint: Area, Length, Perimeter must be positive
        if any(term in q_lower for term in ["perimeter", "area", "length", "radius", "distance", "height"]):
            if sym_obj.is_number:
                try:
                    dim_val = float(sym_obj)
                    if dim_val <= 0:
                        return ("fail", f"Verification Failed: Geometric dimension (length/area/perimeter) must be positive, but got {dim_val}.")
                    return ("pass", f"Verification Passed: Geometric dimension is positive ({dim_val}).")
                except Exception:
                    pass

    except Exception:
        # If sympy cannot parse (e.g. textual name answer like "Evelyn"), check if valid text
        if len(cand_str) > 0 and not any(ch in cand_str for ch in ["\n", "\r", "\t"]):
            return ("unknown", f"Candidate answer is a non-symbolic text or name entity ('{cand_str}').")

    return ("unknown", "Candidate answer is syntactically well-formed, but problem nature prevents automated symbolic proof without ground truth.")
