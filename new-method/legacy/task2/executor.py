#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sandbox executor cho code SymPy do Module 4 sinh ra.

Nhiệm vụ chính: chặn pattern nguy hiểm, chạy code trong thư mục tạm, kiểm tra
stdout chỉ có một `\boxed{...}`, rồi parse answer/unit thô cho verifier.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .verify import extract_unit_from_boxed, latex_to_float

GIVEN_AUDIT_PREFIX = "__GIVEN_AUDIT__"


# ===== Extracted notebook cell 28 =====
# Executes generated Python code in an isolated subprocess.
# Checks for:
#   - Code safety (no file I/O, network, eval, etc.)
#   - Clean boxed output format
#   - Numeric answer extraction from LaTeX

def quick_code_safety_check(code: str) -> Optional[str]:
    """
    Check generated code for unsafe patterns.
    Returns error message if unsafe, None if safe.
    """
    banned_patterns = [
        r"\bimport\s+os\b",
        r"\bimport\s+sys\b",
        r"\bimport\s+subprocess\b",
        r"\bimport\s+socket\b",
        r"\bimport\s+requests\b",
        r"\bopen\s*\(",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"__import__",
        r"\binput\s*\(",
        r"\bcompile\s*\(",
        r"\bassert\b",
        r"\bsp\.Unit\s*\(",
    ]
    for pattern in banned_patterns:
        if re.search(pattern, code):
            return f"Unsafe code pattern detected: {pattern}"
    return None


def extract_all_boxed(stdout: str) -> List[str]:
    """
    Extract all contents from \\boxed{...} in stdout.
    Handles nested braces by tracking depth.
    """
    s = stdout.strip()
    token = r"\boxed{"
    results: List[str] = []
    search_from = 0

    while True:
        start = s.find(token, search_from)
        if start == -1:
            break

        i = start + len(token)
        depth = 1
        chars: List[str] = []

        while i < len(s):
            ch = s[i]
            if ch == "{":
                depth += 1
                chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    results.append("".join(chars).strip())
                    search_from = i + 1
                    break
                chars.append(ch)
            else:
                chars.append(ch)
            i += 1
        else:
            break

    return results


def extract_boxed(stdout: str) -> Optional[str]:
    """
    Extract content from \\boxed{...} in stdout.
    Multiple boxed lines are combined as a semicolon-separated multi-answer.
    """
    boxes = extract_all_boxed(stdout)
    if not boxes:
        return None
    return "; ".join(boxes)


def is_clean_boxed_stdout(stdout: str) -> bool:
    """Check that stdout contains exactly one boxed answer and nothing else."""
    s = stdout.strip()
    if re.fullmatch(r"\\boxed\{.*\}(?:\s*\\boxed\{.*\})*", s, flags=re.DOTALL):
        return True
    return len(extract_all_boxed(s)) == 1 and bool(extract_unit_from_boxed(s))


def split_answer_and_given_audit(stdout: str) -> tuple[str, List[Dict[str, Any]]]:
    answer_lines: List[str] = []
    given_audit: List[Dict[str, Any]] = []
    for line in str(stdout or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(GIVEN_AUDIT_PREFIX):
            payload = stripped[len(GIVEN_AUDIT_PREFIX):].strip()
            try:
                parsed = ast.literal_eval(payload)
                if isinstance(parsed, list):
                    given_audit.extend(x for x in parsed if isinstance(x, dict))
                elif isinstance(parsed, dict):
                    given_audit.append(parsed)
            except Exception:
                given_audit.append({"parse_error": payload})
        elif stripped:
            answer_lines.append(line)
    return "\n".join(answer_lines).strip(), given_audit


def execute_code(code: str, timeout: int) -> Dict[str, Any]:
    """
    Execute Python code in a sandboxed subprocess.

    Returns a dict with:
      - status: "pass" or "fail"
      - error_type: None, "SafetyError", "RuntimeError", "OutputFormatError", "TimeoutExpired"
      - stdout, stderr, boxed, numeric_answer, pred_unit
    """
    # Safety check before execution
    safety_error = quick_code_safety_check(code)
    if safety_error:
        return {
            "status": "fail",
            "error_type": "SafetyError",
            "stdout": "", "stderr": safety_error,
            "boxed": None, "numeric_answer": None, "pred_unit": "",
            "given_audit": [],
        }

    # Execute in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "solution.py"
        script_path.write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            answer_stdout, given_audit = split_answer_and_given_audit(stdout)
            boxed = extract_boxed(answer_stdout)
            numeric_answer = latex_to_float(boxed or answer_stdout)
            pred_unit = extract_unit_from_boxed(boxed or "") or extract_unit_from_boxed(answer_stdout)

            # Check for runtime errors
            if proc.returncode != 0:
                return {
                    "status": "fail", "error_type": "RuntimeError",
                    "returncode": proc.returncode,
                    "stdout": stdout, "stderr": stderr,
                    "boxed": boxed, "numeric_answer": numeric_answer,
                    "pred_unit": pred_unit,
                    "given_audit": given_audit,
                }

            # Check for missing boxed output
            if not boxed:
                return {
                    "status": "fail", "error_type": "OutputFormatError",
                    "returncode": proc.returncode,
                    "stdout": stdout, "stderr": stderr,
                    "boxed": None, "numeric_answer": numeric_answer,
                    "pred_unit": pred_unit,
                    "given_audit": given_audit,
                }

            # Check for clean output (no extra text around \boxed)
            if not is_clean_boxed_stdout(answer_stdout):
                return {
                    "status": "fail", "error_type": "OutputFormatError",
                    "returncode": proc.returncode,
                    "stdout": stdout,
                    "stderr": "stdout must contain exactly one boxed answer",
                    "boxed": boxed, "numeric_answer": numeric_answer,
                    "pred_unit": pred_unit,
                    "given_audit": given_audit,
                }

            # Success!
            return {
                "status": "pass", "error_type": None,
                "returncode": proc.returncode,
                "stdout": stdout, "stderr": stderr,
                "boxed": boxed, "numeric_answer": numeric_answer,
                "pred_unit": pred_unit,
                "given_audit": given_audit,
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "fail", "error_type": "TimeoutExpired",
                "stdout": "", "stderr": f"Timed out after {timeout}s",
                "boxed": None, "numeric_answer": None, "pred_unit": "",
                "given_audit": [],
            }


# ===== Extracted notebook cell 30 =====
# If Module 4 code fails execution, send the code + error back
# to the LLM for fixing. One retry attempt.
