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

    try:
        import ast
        tree = ast.parse(text)
        class RecursionVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_function = None
                self.recursive_calls = []
            def visit_FunctionDef(self, node):
                old_function = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_function
            def visit_Call(self, node):
                if self.current_function and isinstance(node.func, ast.Name):
                    if node.func.id == self.current_function:
                        self.recursive_calls.append(self.current_function)
                self.generic_visit(node)
        visitor = RecursionVisitor()
        visitor.visit(tree)
        if visitor.recursive_calls:
            funcs = ", ".join(repr(f) for f in sorted(set(visitor.recursive_calls)))
            findings.append(f"recursive function call detected in {funcs}; ensure a proper base case exists or rewrite iteratively using a queue/stack to avoid RecursionError")
    except Exception:
        pass

    return findings
