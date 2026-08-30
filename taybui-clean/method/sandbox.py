"""
Module Sandbox thực thi mã nguồn Python/SymPy cách ly và an toàn.
Sử dụng ThreadPoolExecutor để đảm bảo tốc độ thực thi nhanh dưới 0.05 giây và bảo vệ chống treo (timeout).
"""

import io
import json
import math
import os
import queue
import random
import re
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
        "json": json,
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

    if mode in {"symcode", "symplanner"} and sympy is not None:
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

    original_json_dumps = json.dumps

    def _json_dumps_with_default_str(*args, **kwargs):
        kwargs.setdefault("default", str)
        return original_json_dumps(*args, **kwargs)

    try:
        if mode == "symplanner":
            json.dumps = _json_dumps_with_default_str
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, exec_globals)
        stdout_val = stdout_capture.getvalue()
        return {"status": "success", "stdout": stdout_val, "traceback": None}
    except Exception:
        tb_raw = traceback.format_exc()
        tb_clean = _clean_traceback_str(tb_raw)
        return {"status": "error", "stdout": stdout_capture.getvalue(), "traceback": tb_clean}
    finally:
        if mode == "symplanner":
            json.dumps = original_json_dumps


def _read_relaxed_json_value(raw_value: str) -> Any:
    value = raw_value.strip().rstrip(",")
    if not value:
        return None
    if value in {"null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return json.loads(value)
    except Exception:
        return value


def _extract_relaxed_json_field(text: str, key: str) -> Any:
    match = re.search(rf'["\']{re.escape(key)}["\']\s*:', text)
    if not match:
        return None
    idx = match.end()
    while idx < len(text) and text[idx].isspace():
        idx += 1
    start = idx
    quote = None
    depth = 0
    while idx < len(text):
        char = text[idx]
        if quote:
            if char == quote and text[idx - 1] != "\\":
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            if depth == 0 and char == "}":
                break
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            break
        idx += 1
    return _read_relaxed_json_value(text[start:idx])


def _parse_symplanner_structured_stdout(stdout: str) -> Optional[Dict[str, Any]]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None

    candidates = [line for line in lines if line.startswith("{") and "answer" in line]
    if not candidates and len(lines) == 1:
        candidates = [lines[0]]

    for candidate in reversed(candidates):
        structured = None
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                structured = value
        except json.JSONDecodeError:
            if candidate.startswith("{"):
                answer = _extract_relaxed_json_field(candidate, "answer")
                if answer is not None:
                    structured = {
                        "answer": answer,
                        "canonical_answer": _extract_relaxed_json_field(candidate, "canonical_answer"),
                        "answer_type": _extract_relaxed_json_field(candidate, "answer_type"),
                        "unit": _extract_relaxed_json_field(candidate, "unit"),
                        "variables": _extract_relaxed_json_field(candidate, "variables"),
                    }
        if isinstance(structured, dict) and "answer" in structured:
            structured.setdefault("canonical_answer", str(structured.get("answer")))
            structured.setdefault("answer_type", "number")
            structured.setdefault("unit", None)
            structured.setdefault("variables", {})
            if structured.get("variables") is None:
                structured["variables"] = {}
            return structured
    return None


def _run_code_in_process_wrapper(code: str, mode: str, conn) -> None:
    try:
        res = _run_code_in_scope(code, mode)
        conn.send((True, res))
    except Exception as e:
        tb = traceback.format_exc()
        conn.send((False, (str(e), tb)))
    finally:
        conn.close()


def _run_code_with_process_timeout(code: str, mode: str, timeout: float) -> Dict[str, Any]:
    import multiprocessing
    start_method = "spawn"
    if os.name != "nt" and "fork" in multiprocessing.get_all_start_methods():
        start_method = "fork"
    ctx = multiprocessing.get_context(start_method)
    parent_conn, child_conn = ctx.Pipe()
    p = ctx.Process(target=_run_code_in_process_wrapper, args=(code, mode, child_conn))
    p.start()
    
    # Chờ tiến trình con hoàn thành hoặc quá thời gian timeout
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        parent_conn.close()
        return {
            "status": "timeout",
            "stdout": "",
            "traceback": f"Lỗi quá thời gian thực thi (vượt quá {timeout} giây).",
            "extracted_answer": None
        }
    
    if parent_conn.poll():
        try:
            success, val = parent_conn.recv()
        except Exception as e:
            success, val = False, (str(e), traceback.format_exc())
        parent_conn.close()
        if success:
            return val
        else:
            return {
                "status": "error",
                "stdout": "",
                "traceback": val[1],
                "extracted_answer": None
            }
    parent_conn.close()
    return {
        "status": "error",
        "stdout": "",
        "traceback": "Lỗi: Tiến trình con kết thúc đột ngột không phản hồi.",
        "extracted_answer": None
    }


def execute_code_safely(code: str, mode: str = "symcode", timeout: int = 15) -> Dict[str, Any]:
    """
    Thực thi mã nguồn an toàn với cơ chế timeout nghiêm ngặt bằng Process cách ly, cô lập môi trường và trích xuất kết quả.

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

    res = _run_code_with_process_timeout(code, mode, timeout)

    stdout = res.get("stdout", "")
    if mode == "symplanner":
        structured = _parse_symplanner_structured_stdout(stdout)
        if isinstance(structured, dict):
            res["structured_output"] = structured
            res["extracted_answer"] = structured.get("answer")
            res["canonical_answer"] = structured.get("canonical_answer")
            res["answer_type"] = structured.get("answer_type")
            res["unit"] = structured.get("unit")
            res["variables"] = structured.get("variables")
            return res
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if any(line.startswith("{") for line in lines):
            res["status"] = "error"
            res["traceback"] = "Invalid SymPlanner JSON output"
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
