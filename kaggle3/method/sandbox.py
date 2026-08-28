"""
Module Sandbox thực thi mã nguồn Python/SymPy cách ly và an toàn.
Sử dụng multiprocessing.Process để đảm bảo cô lập hoàn toàn tiến trình và cưỡng chế diệt (terminate/kill)
ngay lập tức nếu đoạn mã của LLM bị lặp vô hạn (infinite loop) hoặc vượt quá timeout.
"""

import io
import math
import traceback
import contextlib
import multiprocessing
from typing import Dict, Any, Optional

from .extractor import extract_boxed_content

# Tải trước các thư viện khoa học và toán biểu tượng
try:
    import sympy
    import sympy as sp
    from sympy import (
        symbols, Symbol, Eq, solve, solveset, linsolve, nonlinsolve,
        simplify, expand, factor, Rational, Integer, Float,
        sympify, factorint, primefactors, isprime, nextprime,
        pi, sqrt, root, sin, cos, tan, atan2, Matrix,
        Point, Line, Circle, Polygon, Segment, Ray,
        binomial, factorial, fibonacci, Sum, Product, limit, diff, integrate,
        zoo, oo, nan
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

try:
    import collections
except ImportError:
    collections = None

try:
    import numpy as np
except ImportError:
    np = None


def _clean_traceback_str(tb_str: str) -> str:
    """
    Lọc bỏ các frame nội bộ của sandbox, chỉ giữ lại các dòng liên quan trực tiếp đến script của mô hình.
    """
    lines = tb_str.strip().split("\n")
    relevant_lines = []
    capture = False
    for line in lines:
        if 'File "<string>"' in line:
            capture = True
        if capture:
            relevant_lines.append(line)
            
    if relevant_lines:
        return "\n".join(relevant_lines)
    return "\n".join(lines[-5:]) if len(lines) > 5 else tb_str


def _run_code_in_scope(code: str, mode: str = "symcode") -> Dict[str, Any]:
    """
    Thực thi mã nguồn Python/SymPy trong phạm vi biến cô lập và thu thập stdout / traceback.
    """
    import math, random
    stdout_capture = io.StringIO()
    
    exec_globals = {
        "__builtins__": __builtins__,
        "math": math,
        "random": random,
    }
    
    if fractions is not None:
        exec_globals["fractions"] = fractions
        exec_globals["Fraction"] = Fraction
        
    if itertools is not None:
        exec_globals["itertools"] = itertools

    if collections is not None:
        exec_globals["collections"] = collections

    if np is not None:
        exec_globals["np"] = np
        exec_globals["numpy"] = np

    if mode == "symcode" and sympy is not None:
        exec_globals.update({
            "sympy": sympy,
            "sp": sympy,
            "symbols": symbols,
            "Symbol": Symbol,
            "Eq": Eq,
            "solve": solve,
            "solveset": solveset,
            "linsolve": linsolve,
            "nonlinsolve": nonlinsolve,
            "simplify": simplify,
            "expand": expand,
            "factor": factor,
            "Rational": Rational,
            "Integer": Integer,
            "Float": Float,
            "sympify": sympify,
            "factorint": factorint,
            "primefactors": primefactors,
            "isprime": isprime,
            "nextprime": nextprime,
            "pi": pi,
            "sqrt": sqrt,
            "root": root,
            "sin": sin,
            "cos": cos,
            "tan": tan,
            "atan2": atan2,
            "Matrix": Matrix,
            "binomial": binomial,
            "factorial": factorial,
            "fibonacci": fibonacci,
            "Sum": Sum,
            "Product": Product,
            "limit": limit,
            "diff": diff,
            "integrate": integrate,
            "zoo": zoo,
            "oo": oo,
            "nan": nan
        })

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, exec_globals)
        stdout_val = stdout_capture.getvalue()
        return {"status": "success", "stdout": stdout_val, "traceback": None}
    except Exception:
        tb_raw = traceback.format_exc()
        tb_clean = _clean_traceback_str(tb_raw)
        return {"status": "error", "stdout": stdout_capture.getvalue(), "traceback": tb_clean}


def _process_sandbox_target(code: str, mode: str, q: multiprocessing.Queue):
    """Worker target chạy trong tiến trình riêng cô lập."""
    res = _run_code_in_scope(code, mode)
    q.put(res)


def execute_code_safely(code: str, mode: str = "symcode", timeout: int = 15) -> Dict[str, Any]:
    """
    Thực thi mã nguồn an toàn với cơ chế Process isolation & Hard Timeout termination.
    Nếu đoạn mã bị lặp vô hạn (while True), tiến trình con sẽ bị diệt (terminate/kill) ngay lập tức.
    """
    if not code or not code.strip():
        return {
            "status": "error",
            "stdout": "",
            "traceback": "Lỗi: Không tìm thấy đoạn mã Python hợp lệ để thực thi.",
            "extracted_answer": None
        }

    ctx = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else multiprocessing.get_context()
    q = ctx.Queue()
    p = ctx.Process(target=_process_sandbox_target, args=(code, mode, q))
    
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join(timeout=1)
        if p.is_alive():
            p.kill()
        return {
            "status": "timeout",
            "stdout": "",
            "traceback": f"Lỗi quá thời gian thực thi (vượt quá {timeout} giây do lặp vô hạn hoặc tính toán quá lâu).",
            "extracted_answer": None
        }

    if not q.empty():
        res = q.get()
    else:
        res = {
            "status": "error",
            "stdout": "",
            "traceback": "Tiến trình thực thi mã nguồn kết thúc bất ngờ mà không có kết quả.",
            "extracted_answer": None
        }

    stdout = res.get("stdout", "")
    boxed_ans = extract_boxed_content(stdout)
    if boxed_ans is not None:
        res["extracted_answer"] = boxed_ans
    else:
        lines = [l.strip() for l in stdout.strip().split("\n") if l.strip()]
        res["extracted_answer"] = lines[-1] if lines else None

    return res
