"""Cheap, deterministic diagnostics for common generated SymPy failure modes."""

from __future__ import annotations

import re


def lint_sympy_code(code: str) -> list[str]:
    text = str(code or "")
    findings: list[str] = []
    patterns = [
        (r"(?:solve|solveset|nonlinsolve|linsolve)\([^\n]+\)\s*\[\s*0\s*\]", "indexing a solver result without checking for an empty solution"),
        (r"\b(?:solutions?|roots?|sol)\s*\[\s*0\s*\]", "indexing a possibly empty solution list"),
        (r"(?:Eq|solve)\([^\n]*//", "using floor division inside an equation/solver expression"),
        (r"\)\s*&\s*\(", "combining inequalities with Python '&' instead of SymPy And"),
        (r"\.evalf\(\)", "evalf() may be called on a Python scalar; use sp.sympify first"),
        (r"print\(\s*['\"]\{.*['\"]\.format\(", "hand-formatting JSON with .format; use json.dumps(..., default=str)"),
    ]
    for pattern, message in patterns:
        if re.search(pattern, text):
            findings.append(message)
    return findings
