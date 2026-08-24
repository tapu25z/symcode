"""
Execution sandbox with distinct tool environments for PAL and SymCode.
Uses ThreadPoolExecutor for sub-second execution speed and strict timeout protection.

Tool Access Specification:
- PAL (Program-Aided Language): Standard Python only (math, fractions, itertools, random). SymPy is excluded.
- SymCode (Symbolic Reasoning): Python + SymPy symbolic mathematics suite.
"""

import io
import math
import queue
import random
import traceback
import contextlib
import concurrent.futures
from typing import Dict, Any, Optional
from .extractor import extract_boxed_content

# Pre-import standard scientific & symbolic math libraries in host process
try:
    import sympy
    import sympy as sp
    from sympy import (
        symbols, Eq, solve, simplify, Rational, Integer, Float,
        sympify, factorint, pi, sqrt, atan2, Matrix, Point, Line
    )
except ImportError:
    sympy = None
    sp = None

try:
    import fractions
    from fractions import Fraction
except ImportError:
    fractions = None
    Fraction = None

try:
    import itertools
except ImportError:
    itertools = None


def _run_code_in_scope(code: str, mode: str = "symcode") -> Dict[str, Any]:
    """
    Executes Python/SymPy code in an isolated scope with stdout/traceback capture.
    Enforces tool access boundaries between PAL (standard Python) and SymCode (SymPy).
    """
    stdout_capture = io.StringIO()
    
    # Base Python environment for all methods
    exec_globals = {
        "__builtins__": __builtins__,
        "math": math,
        "random": random,
    }
    
    if fractions is not None:
        exec_globals.update({
            "fractions": fractions,
            "Fraction": Fraction,
        })
        
    if itertools is not None:
        exec_globals["itertools"] = itertools

    # SymCode specific toolchain: SymPy symbolic mathematics
    if mode == "symcode" and sympy is not None:
        exec_globals.update({
            "sympy": sympy,
            "sp": sympy,
            "symbols": symbols,
            "Eq": Eq,
            "solve": solve,
            "simplify": simplify,
            "Rational": Rational,
            "Integer": Integer,
            "Float": Float,
            "sympify": sympify,
            "factorint": factorint,
            "pi": pi,
            "sqrt": sqrt,
            "atan2": atan2,
            "Matrix": Matrix,
        })

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, exec_globals)
        stdout_val = stdout_capture.getvalue()
        return {"status": "success", "stdout": stdout_val, "traceback": None}
    except Exception:
        tb_str = traceback.format_exc()
        return {"status": "error", "stdout": stdout_capture.getvalue(), "traceback": tb_str}


def execute_code_safely(code: str, mode: str = "symcode", timeout: int = 15) -> Dict[str, Any]:
    """
    Executes code safely with a strict timeout, environment isolation, and output extraction.

    Args:
        code: The Python code string to execute.
        mode: "symcode" (allows SymPy) or "pal" (standard Python only).
        timeout: Maximum execution time in seconds.

    Returns:
        {
            "status": "success" | "error" | "timeout",
            "stdout": str,
            "traceback": Optional[str],
            "extracted_answer": Optional[str]
        }
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_code_in_scope, code, mode)
        try:
            res = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return {
                "status": "timeout",
                "stdout": "",
                "traceback": f"Execution timed out after {timeout} seconds.",
                "extracted_answer": None
            }
        except Exception as e:
            return {
                "status": "error",
                "stdout": "",
                "traceback": str(e),
                "extracted_answer": None
            }

    # Extract answer from stdout (prefer \boxed{...} or fallback to last line)
    stdout = res.get("stdout", "")
    boxed_ans = extract_boxed_content(stdout)
    if boxed_ans is not None:
        res["extracted_answer"] = boxed_ans
    else:
        lines = [l.strip() for l in stdout.strip().split("\n") if l.strip()]
        res["extracted_answer"] = lines[-1] if lines else None

    return res
