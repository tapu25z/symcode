"""
Module Sandbox thực thi mã nguồn Python/SymPy cách ly và an toàn.
Sử dụng ThreadPoolExecutor để đảm bảo tốc độ thực thi nhanh dưới 0.05 giây và bảo vệ chống treo (timeout).
"""

import io
import json
import math
import queue
import random
import traceback
import contextlib
import concurrent.futures
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


def execute_code_safely(code: str, mode: str = "symcode", timeout: int = 15) -> Dict[str, Any]:
    """
    Thực thi mã nguồn an toàn với cơ chế timeout nghiêm ngặt, cô lập môi trường và trích xuất kết quả.

    Args:
        code: Chuỗi mã nguồn Python cần chạy.
        mode: "symcode" (boxed output), "symplanner" (structured JSON output), hoặc "pal" (chỉ Python chuẩn).
        timeout: Thời gian thực thi tối đa tính bằng giây.

    Returns:
        Dict gồm các trường:
            - "status": "success" | "error" | "timeout"
            - "stdout": chuỗi đầu ra tiêu chuẩn
            - "traceback": thông báo lỗi chi tiết (nếu có)
            - "extracted_answer": đáp án trích xuất được từ stdout
    """
    if not code or not code.strip():
        return {
            "status": "error",
            "stdout": "",
            "traceback": "Lỗi: Không tìm thấy đoạn mã Python hợp lệ để thực thi.",
            "extracted_answer": None
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_code_in_scope, code, mode)
        try:
            res = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return {
                "status": "timeout",
                "stdout": "",
                "traceback": f"Lỗi quá thời gian thực thi (vượt quá {timeout} giây).",
                "extracted_answer": None
            }
        except Exception as e:
            return {
                "status": "error",
                "stdout": "",
                "traceback": str(e),
                "extracted_answer": None
            }

    stdout = res.get("stdout", "")
    if mode == "symplanner":
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if len(lines) == 1:
            try:
                structured = json.loads(lines[0])
            except json.JSONDecodeError:
                structured = None
            if isinstance(structured, dict) and "answer" in structured:
                required = {"answer", "canonical_answer", "answer_type", "unit", "variables"}
                missing = sorted(required - set(structured))
                if missing:
                    res["status"] = "error"
                    res["traceback"] = f"Structured output missing fields: {', '.join(missing)}"
                    res["extracted_answer"] = None
                    return res
                res["structured_output"] = structured
                res["extracted_answer"] = structured.get("answer")
                res["canonical_answer"] = structured.get("canonical_answer")
                res["answer_type"] = structured.get("answer_type")
                res["unit"] = structured.get("unit")
                res["variables"] = structured.get("variables")
                return res
            if lines[0].startswith("{"):
                res["status"] = "error"
                res["traceback"] = "Invalid SymPlanner JSON output"
                res["extracted_answer"] = None
                return res
        elif any(line.startswith("{") for line in lines):
            res["status"] = "error"
            res["traceback"] = "SymPlanner output must contain exactly one JSON line"
            res["extracted_answer"] = None
            return res
        # Backward-compatible fallback for old SymPlanner checkpoints/code.
    boxed_ans = extract_boxed_content(stdout)
    if boxed_ans is not None:
        res["extracted_answer"] = boxed_ans
    else:
        lines = [l.strip() for l in stdout.strip().split("\n") if l.strip()]
        res["extracted_answer"] = lines[-1] if lines else None

    return res
