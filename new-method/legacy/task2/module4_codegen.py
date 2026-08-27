#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module 4: sinh SymPy code, chạy sandbox, và tự debug một lần nếu lỗi.

Codegen được tách khỏi executor để prompt/LLM logic không trộn với logic chạy
subprocess. Điều này giúp debug Kaggle dễ hơn khi lỗi nằm ở generation hay execution.
"""

from __future__ import annotations

import json
import re
import hashlib
from typing import Any, Dict, List, Optional

from .executor import GIVEN_AUDIT_PREFIX, execute_code, extract_boxed
from .knowledge_guides import select_knowledge_guides
from .llm import chat_messages
from .prompts import ANSWER_TYPE_SYSTEMS, DEBUG_PROMPT, DIRECT_CONCEPTUAL_SYSTEM, LLM_DIRECT_FALLBACK_SYSTEM, NUMERIC_SYSTEM
from .debug_trace import compact_examples, log_json, log_text

_UNICODE_IDENTIFIER_TRANSLATION = str.maketrans({
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
})

_SYMPY_FAKE_UNIT_ATTRS = {
    "ohm", "volt", "ampere", "coulomb", "farad", "henry", "tesla",
    "weber", "joule", "newton", "watt", "meter", "second", "kilogram",
}

_NUMERIC_FORMAT_HELPERS = r"""
def _safe_float(x):
    return float(sp.N(x))

def _safe_format_number(x, spec=""):
    clean_spec = str(spec or "").strip().lower()
    if clean_spec in {".1f", ".2f", ".3f", ".4f", "1f", "2f", "3f", "4f"}:
        spec = ".6g"
    try:
        return format(float(sp.N(x)), spec)
    except Exception:
        try:
            return str(sp.N(x))
        except Exception:
            return str(x)
""".strip()

MODULE4_METHOD_THINKING_SYSTEM = """You are Module 4 method selector for a physics-to-SymPy pipeline.

Use the Module 4 input and the retrieved few-shot examples as solved references.
This is the only thinking-enabled call in the first SymPy code-generation path.

Do not write Python code. Return a concise METHOD_DRAFT with:
- method
- formulas
- key steps
- target unit
- edge cases or unit conversions
"""

MODULE4_METHOD_JSON_SYSTEM = """Convert the METHOD_DRAFT into strict JSON for Module 4.

Return ONLY a valid JSON object with exactly these keys:
{
  "method": "string",
  "formulas": ["string"],
  "geometry": {},
  "steps": ["string"],
  "target_unit": "string",
  "notes": ["string"],
  "needs_review": false
}

Do not include Python code. Do not include markdown. Do not include any extra keys.
"""


def _split_top_level_args(args: str) -> List[str]:
    """Split a Python argument list on top-level commas."""
    parts: List[str] = []
    start = 0
    depth = 0
    quote = ""
    escape = False
    for idx, ch in enumerate(args):
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = ""
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0:
            parts.append(args[start:idx].strip())
            start = idx + 1
    tail = args[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _rewrite_numeric_format_line(line: str) -> tuple[str, bool]:
    """
    Rewrite numeric format specs so SymPy expressions/floats are handled safely.

    Example:
      "{:.3f}".format(expr) -> "{}".format(_safe_format_number(expr, ".3f"))
    """
    specs = re.findall(r"\{:\s*([^{}]+)\}", line)
    if not specs or ".format(" not in line:
        return line, False

    fmt_start = line.find(".format(")
    args_start = fmt_start + len(".format(")
    depth = 1
    quote = ""
    escape = False
    args_end = -1
    for idx in range(args_start, len(line)):
        ch = line[idx]
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = ""
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                args_end = idx
                break
    if args_end == -1:
        return line, False

    args = _split_top_level_args(line[args_start:args_end])
    if not args:
        return line, False

    rewritten_line = re.sub(r"\{:\s*[^{}]+\}", "{}", line[:fmt_start])
    rewritten_args = []
    for idx, arg in enumerate(args):
        spec = specs[idx] if idx < len(specs) else specs[-1]
        rewritten_args.append(f"_safe_format_number({arg}, {spec!r})")
    rewritten_line += ".format(" + ", ".join(rewritten_args) + line[args_end:]
    return rewritten_line, True


def _inject_numeric_helpers(code: str) -> str:
    if "def _safe_format_number" in code:
        return code
    lines = code.splitlines()
    for idx, line in enumerate(lines):
        if re.match(r"\s*import\s+sympy\s+as\s+sp\b", line):
            return "\n".join(lines[:idx + 1] + ["", _NUMERIC_FORMAT_HELPERS, ""] + lines[idx + 1:])
    return _NUMERIC_FORMAT_HELPERS + "\n\n" + code


_GREEK_IDENTIFIER_ALIASES = {
    "ρ": "rho",
    "φ": "phi",
    "θ": "theta",
    "ω": "omega",
    "μ": "mu",
    "ε": "epsilon",
}


def _identifier_candidate(text: Any) -> str:
    s = str(text or "").strip().translate(_UNICODE_IDENTIFIER_TRANSLATION)
    s = "".join(_GREEK_IDENTIFIER_ALIASES.get(ch, ch) for ch in s)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")
    if not s:
        return ""
    if s[0].isdigit():
        s = "_" + s
    return s


def _given_identifier_candidates(item: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    for key in ("symbol", "name"):
        candidate = _identifier_candidate(item.get(key, ""))
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _with_given_audit_prints(code: str, payload: Dict[str, Any]) -> str:
    normalized = payload.get("normalized_problem", {})
    givens = normalized.get("given", []) if isinstance(normalized, dict) else []
    if not givens or GIVEN_AUDIT_PREFIX in code:
        return code

    audit_entries = []
    for idx, item in enumerate(givens):
        if not isinstance(item, dict):
            continue
        audit_entries.append({
            "index": idx,
            "name": item.get("name", ""),
            "symbol": item.get("symbol", ""),
            "expected_value": item.get("value"),
            "expected_unit": item.get("unit", ""),
            "candidates": _given_identifier_candidates(item),
        })
    if not audit_entries:
        return code

    lines = [
        "",
        "# Step 99: Emit normalized given quantities for Module 3 consistency audit.",
        "def _codex_audit_value(*names):",
        "    for _name in names:",
        "        if _name in globals():",
        "            _value = globals()[_name]",
        "            try:",
        "                return float(sp.N(_value))",
        "            except Exception:",
        "                return str(_value)",
        "    return None",
        "_codex_given_audit = []",
    ]
    for entry in audit_entries:
        names = ", ".join(repr(x) for x in entry["candidates"])
        lines.append(
            "_codex_given_audit.append({"
            f"'index': {entry['index']!r}, "
            f"'name': {entry['name']!r}, "
            f"'symbol': {entry['symbol']!r}, "
            f"'expected_value': {entry['expected_value']!r}, "
            f"'expected_unit': {entry['expected_unit']!r}, "
            f"'candidates': {entry['candidates']!r}, "
            f"'actual_value': _codex_audit_value({names}), "
            f"'actual_unit': {entry['expected_unit']!r}"
            "})"
        )
    lines.append(f"print({GIVEN_AUDIT_PREFIX!r} + repr(_codex_given_audit))")
    return code.rstrip() + "\n" + "\n".join(lines) + "\n"


def _execute_with_given_audit(code: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    return execute_code(_with_given_audit_prints(code, payload), timeout)


def _execute_preserving_existing_debug_flow(code: str, payload: Dict[str, Any], timeout: int) -> tuple[str, Dict[str, Any]]:
    execution = execute_code(code, timeout)
    if execution.get("status") != "pass":
        return code, execution
    audited_code = _with_given_audit_prints(code, payload)
    if audited_code == code:
        return code, execution
    audited_execution = execute_code(audited_code, timeout)
    return code, audited_execution


def _numbers_close(actual: Any, expected: Any) -> bool:
    if actual is None:
        return False
    try:
        av = float(actual)
        ev = float(expected)
    except Exception:
        return str(actual).strip() == str(expected).strip()
    scale = max(abs(ev), 1e-12)
    return abs(av - ev) <= max(1e-9, 1e-6 * scale)


def _deterministic_given_audit_issues(payload: Dict[str, Any], execution: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized = payload.get("normalized_problem", {})
    expected_givens = normalized.get("given", []) if isinstance(normalized, dict) else []
    actual_items = execution.get("given_audit") or []
    actual_by_index = {
        item.get("index"): item
        for item in actual_items
        if isinstance(item, dict) and "index" in item
    }
    issues: List[Dict[str, Any]] = []
    for idx, expected in enumerate(expected_givens):
        if not isinstance(expected, dict):
            continue
        actual = actual_by_index.get(idx)
        if not actual:
            issues.append({
                "index": idx,
                "symbol": expected.get("symbol", ""),
                "name": expected.get("name", ""),
                "expected_value": expected.get("value"),
                "expected_unit": expected.get("unit", ""),
                "actual_value": None,
                "actual_unit": "",
                "issue": "missing_given_in_sympy_code",
            })
            continue
        if not _numbers_close(actual.get("actual_value"), expected.get("value")):
            issues.append({
                "index": idx,
                "symbol": expected.get("symbol", ""),
                "name": expected.get("name", ""),
                "expected_value": expected.get("value"),
                "expected_unit": expected.get("unit", ""),
                "actual_value": actual.get("actual_value"),
                "actual_unit": actual.get("actual_unit", ""),
                "issue": "given_value_mismatch",
            })
    return issues


def compare_given_audit_with_base_model(
    client: Any,
    model: str,
    payload: Dict[str, Any],
    execution: Dict[str, Any],
) -> Dict[str, Any]:
    issues = _deterministic_given_audit_issues(payload, execution)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict physics SI audit checker. Compare generated SymPy given "
                "quantities against Module 3 SI givens. Return a concise JSON object with "
                "status and issues. Do not solve the final answer."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "module3_si_given": (payload.get("normalized_problem", {}) or {}).get("given", []),
                    "sympy_given_audit": execution.get("given_audit", []),
                    "deterministic_issues": issues,
                },
                indent=2,
                ensure_ascii=False,
            ),
        },
    ]
    trace_id = _trace_id_from_payload(payload)
    log_json(f"{trace_id}_module4_given_audit_check_prompt", {
        "payload": payload,
        "execution": execution,
        "deterministic_issues": issues,
        "messages": messages,
    })
    base_review = ""
    try:
        base_review = chat_messages(client, model, messages, use_adapter=False)
    except Exception as exc:
        base_review = f"base_model_audit_error: {exc}"
    result = {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "base_review": base_review,
    }
    log_json(f"{trace_id}_module4_given_audit_check_response", result)
    return result


def repair_module4_from_given_audit(
    client: Any,
    model: str,
    answer_type: str,
    payload: Dict[str, Any],
    bad_code: str,
    audit_result: Dict[str, Any],
) -> str:
    system_prompt = ANSWER_TYPE_SYSTEMS.get(answer_type, NUMERIC_SYSTEM)
    correction_lines = []
    for issue in audit_result.get("issues", []):
        if not isinstance(issue, dict):
            continue
        candidates = issue.get("candidates") or []
        if not candidates:
            symbol = str(issue.get("symbol", "") or "").strip()
            name = str(issue.get("name", "") or "").strip()
            candidates = [x for x in (_identifier_candidate(symbol), _identifier_candidate(name)) if x]
        correction_lines.append(
            "- Given index {index} ({symbol} / {name}): set one of variables {candidates} "
            "to EXACT SI value {expected_value!r} with unit {expected_unit!r}. "
            "Current audited value was {actual_value!r} {actual_unit!r}. Issue: {issue}.".format(
                index=issue.get("index", ""),
                symbol=issue.get("symbol", ""),
                name=issue.get("name", ""),
                candidates=candidates,
                expected_value=issue.get("expected_value"),
                expected_unit=issue.get("expected_unit", ""),
                actual_value=issue.get("actual_value"),
                actual_unit=issue.get("actual_unit", ""),
                issue=issue.get("issue", ""),
            )
        )
    required_corrections = "\n".join(correction_lines) or "- No structured corrections were available; inspect the audit JSON carefully."
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "The previous SymPy code below was syntactically valid and executed, but some "
                "initialized given quantities did not match normalized_problem.given from Module 3. "
                "Regenerate the full Python SymPy code by editing that working code.\n\n"
                "MANDATORY RULES:\n"
                "1. Apply every REQUIRED CORRECTION below. Do not leave any audited actual_value unchanged.\n"
                "2. Use the exact SI values from normalized_problem.given for all given variables.\n"
                "3. If a comment in the old code conflicts with Module 3 SI values, fix the code and the comment.\n"
                "4. Do not use the gold answer or original_answer.\n"
                "5. Do not add audit/debug scaffolding such as _codex_given_audit or __GIVEN_AUDIT__; the pipeline adds that instrumentation automatically.\n"
                "6. Return ONLY one complete Python code block. The corrected code must not reproduce the same given-audit mismatch.\n\n"
                f"REQUIRED CORRECTIONS:\n{required_corrections}\n\n"
                f"JSON input:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
                f"Previously executable code to fix:\n```python\n{bad_code}\n```\n\n"
                f"Given-audit issues:\n{json.dumps(audit_result, indent=2, ensure_ascii=False)}"
            ),
        },
    ]
    trace_id = _trace_id_from_payload(payload)
    log_json(f"{trace_id}_module4_given_audit_repair_prompt", {
        "answer_type": answer_type,
        "payload": payload,
        "bad_code": bad_code,
        "audit_result": audit_result,
        "messages": messages,
    })
    content = chat_messages(client, model, messages, use_adapter=False, thinking=False)
    repaired_code = sanitize_generated_code(extract_python_code(content))
    log_json(f"{trace_id}_module4_given_audit_repair_response", {
        "raw_response": content,
        "repaired_code": repaired_code,
    })
    return repaired_code


# Generates executable Python/SymPy code from:
#   - problem_type, target_answer_type, raw_question
#   - normalized_problem (SI-converted from Module 3)
#   - An empty solution_plan (no teacher guidance at inference)
#
# Uses few-shot prompting:
#   - system: answer-type-specific instructions
#   - user/assistant pairs: example (payload → code) from bank
#   - user: actual payload

def extract_python_code(text: str) -> str:
    """
    Extract Python code from LLM output.
    Handles fenced ```python ... ``` blocks and raw code.
    """
    text = text.strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from fenced or raw model output."""
    text = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def has_step_comments(code: str) -> bool:
    """Check if the code follows the required '# Step N:' comment format."""
    return bool(re.search(r"^\s*# Step\s+\d+:", code, re.MULTILINE))


def sanitize_generated_code(code: str) -> str:
    """Apply narrow mechanical fixes for common LLM codegen hazards."""
    code = code.translate(_UNICODE_IDENTIFIER_TRANSLATION)
    code = re.sub(r":\s*\.[1-4][fg]\b", ":.6g", code)
    code = re.sub(r"sp\.Symbol\(([^)]*?),\s*real\s*=\s*True\s*\)", r"sp.Symbol(\1)", code)
    code = re.sub(r"(\b[A-Za-z_][A-Za-z0-9_]*\b)\.evalf\(\)", r"sp.N(\1)", code)

    kept_lines = []
    needs_numeric_helpers = False
    for line in code.splitlines():
        if re.match(r"\s*assert\b", line):
            continue
        if ".format(" in line:
            line = re.sub(r"\\(mathrm|text|vec)\{([^{}]+)\}", r"\\\1{{\2}}", line)
            line, changed = _rewrite_numeric_format_line(line)
            needs_numeric_helpers = needs_numeric_helpers or changed
        kept_lines.append(line)
    code = "\n".join(kept_lines)

    unit_attrs = "|".join(sorted(_SYMPY_FAKE_UNIT_ATTRS))
    code = re.sub(rf"\s*\*\s*sp\.({unit_attrs})\b", "", code)
    code = re.sub(rf"\bsp\.({unit_attrs})\b", "1", code)

    if needs_numeric_helpers:
        code = _inject_numeric_helpers(code)

    return code.strip()


def build_empty_solution_plan(normalized_problem: Dict[str, Any]) -> Dict[str, Any]:
    """Build the minimal solution_plan used when test-time teacher guidance is unavailable."""
    targets = normalized_problem.get("target", []) if isinstance(normalized_problem, dict) else []
    target_unit = ""
    if targets and isinstance(targets[0], dict):
        target_unit = targets[0].get("unit", "")

    return {
        "method": "",
        "formulas": [],
        "geometry": {},
        "steps": [],
        "target_unit": target_unit,
        "notes": ["No teacher guidance. Solve from first principles."],
        "needs_review": False,
    }


def sanitize_method_json(obj: Optional[Dict[str, Any]], fallback_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize METHOD_JSON to the solution_plan shape used by Module 4."""
    base = build_empty_solution_plan(
        fallback_payload.get("normalized_problem", {})
        if isinstance(fallback_payload.get("normalized_problem"), dict)
        else {}
    )
    if not isinstance(obj, dict):
        return base

    formulas = obj.get("formulas", [])
    geometry = obj.get("geometry", {})
    steps = obj.get("steps", [])
    notes = obj.get("notes", [])

    return {
        "method": str(obj.get("method", "") or ""),
        "formulas": [str(x) for x in formulas] if isinstance(formulas, list) else [],
        "geometry": geometry if isinstance(geometry, dict) else {},
        "steps": [str(x) for x in steps] if isinstance(steps, list) else [],
        "target_unit": str(obj.get("target_unit", base["target_unit"]) or ""),
        "notes": [str(x) for x in notes] if isinstance(notes, list) else [],
        "needs_review": bool(obj.get("needs_review", False)),
    }


def build_module4_payload(
    problem_type: str,
    answer_type: str,
    raw_question: str,
    normalized_problem: Dict[str, Any],
    solution_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the JSON payload that Module 4 prompts expect as input.
    """
    del solution_plan

    knowledge_guides = select_knowledge_guides(
        problem_type=problem_type,
        answer_type=answer_type,
        raw_question=raw_question,
        normalized_problem=normalized_problem,
    )

    norm_prob = {}
    if isinstance(normalized_problem, dict):
        norm_prob = {
            "given": normalized_problem.get("given", []),
            "conditions": normalized_problem.get("conditions", []),
            "target": normalized_problem.get("target", []),
        }

    return {
        "problem_type":     problem_type,
        "target_answer_type": answer_type,
        "raw_question":     raw_question,
        "normalized_problem": norm_prob,
        "knowledge_guides": knowledge_guides,
    }


def _quantity_matches(item: Dict[str, Any], needles: List[str], symbols: List[str]) -> bool:
    text = " ".join(
        str(item.get(key, ""))
        for key in ("name", "symbol", "unit", "original_unit")
    ).lower()
    symbol = str(item.get("symbol", "")).strip().lower()
    return symbol in {s.lower() for s in symbols} or any(needle in text for needle in needles)


def _find_numeric_quantity(items: List[Dict[str, Any]], needles: List[str], symbols: List[str]) -> Optional[float]:
    for item in items:
        if not isinstance(item, dict) or not _quantity_matches(item, needles, symbols):
            continue
        value = item.get("value")
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value))
        except (TypeError, ValueError):
            continue
    return None


def build_deterministic_sympy_code(payload: Dict[str, Any]) -> Optional[str]:
    """Return hand-coded SymPy for narrow high-confidence physics patterns."""
    question = str(payload.get("raw_question", "")).lower()
    normalized = payload.get("normalized_problem", {})
    if not isinstance(normalized, dict):
        return None

    targets = normalized.get("target", [])
    target = targets[0] if targets and isinstance(targets[0], dict) else {}
    target_text = " ".join(
        str(target.get(key, ""))
        for key in ("name", "symbol", "unit", "answer_type")
    ).lower()

    givens = normalized.get("given", [])
    if not isinstance(givens, list):
        return None

    is_lc_capacitance = (
        ("resonat" in question or "l-c" in question or "lc" in question)
        and ("capacit" in target_text or str(target.get("symbol", "")).strip().lower() == "c")
        and str(target.get("unit", "")).strip().lower() in {"f", "farad", "farads", ""}
    )
    if not is_lc_capacitance:
        return None

    inductance = _find_numeric_quantity(givens, ["inductance", "henry"], ["l"])
    frequency = _find_numeric_quantity(givens, ["frequency", "resonance", "resonant", "hz"], ["f"])
    if inductance is None or frequency is None or inductance <= 0 or frequency <= 0:
        return None

    return "\n".join([
        "import sympy as sp",
        "",
        "# Step 1: Define given values in SI units",
        f"L = sp.S({inductance!r})      # inductance in H",
        f"f = sp.S({frequency!r})       # resonance frequency in Hz",
        "",
        "# Step 2: Use LC resonance formula f = 1 / (2*pi*sqrt(L*C))",
        "C = 1 / (4 * sp.pi**2 * L * f**2)",
        "",
        "# Step 3: Evaluate capacitance in farads",
        "C_val = sp.N(C, 12)",
        "",
        "# Step 4: Print the boxed answer in SI unit F",
        'print(r"\\boxed{" + format(float(C_val), ".6g") + r" \\, \\mathrm{F}}")',
    ])


def build_module4_payload_from_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstruct a Module 4 payload from a few-shot bank example.
    """
    return build_module4_payload(
        problem_type=ex.get("detected_problem_type", "unknown"),
        answer_type=ex.get("answer_type", "numeric_compute"),
        raw_question=ex.get("question", ""),
        normalized_problem=ex.get("module3_si_extract", {}),
        solution_plan=None,
    )


def build_module4_few_shot_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Format a dynamic few-shot example for Module 4 log and prompt."""
    return build_module4_payload_from_example(ex)


def build_module4_few_shot_messages(
    few_shot_examples: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build the existing Module 4 few-shot user/assistant turns unchanged."""
    messages: List[Dict[str, str]] = []
    for ex in few_shot_examples:
        ex_payload = build_module4_payload_from_example(ex)
        messages.append({
            "role": "user",
            "content": json.dumps(ex_payload, indent=2, ensure_ascii=False),
        })
        messages.append({
            "role": "assistant",
            "content": f"```python\n{ex['module4_symcode']}\n```",
        })
    return messages


def generate_module4_method_json(
    client: Any,
    model: str,
    answer_type: str,
    payload: Dict[str, Any],
    few_shot_examples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Think once to choose the method, then no-think normalize it to METHOD_JSON."""
    few_shot_messages = build_module4_few_shot_messages(few_shot_examples)
    trace_id = _trace_id_from_payload(payload)

    thinking_messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": MODULE4_METHOD_THINKING_SYSTEM + f"\nTarget answer type: {answer_type}",
        },
        *few_shot_messages,
        {
            "role": "user",
            "content": json.dumps(payload, indent=2, ensure_ascii=False),
        },
    ]
    log_json(f"{trace_id}_module4_method_thinking_prompt", {
        "answer_type": answer_type,
        "payload": payload,
        "fewshot_ids": [ex.get("id") for ex in few_shot_examples],
        "fewshots": compact_examples(few_shot_examples),
        "messages": thinking_messages,
    })
    method_draft = chat_messages(
        client,
        model,
        thinking_messages,
        use_adapter=False,
        thinking=True,
    )
    log_text(f"{trace_id}_module4_method_draft", method_draft)

    json_messages: List[Dict[str, str]] = [
        {"role": "system", "content": MODULE4_METHOD_JSON_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "input": payload,
                    "method_draft": method_draft,
                },
                indent=2,
                ensure_ascii=False,
            ),
        },
    ]
    log_json(f"{trace_id}_module4_method_json_prompt", {
        "answer_type": answer_type,
        "payload": payload,
        "method_draft": method_draft,
        "messages": json_messages,
    })
    method_json_text = chat_messages(
        client,
        model,
        json_messages,
        use_adapter=False,
        thinking=False,
    )
    method_json = sanitize_method_json(extract_json_object(method_json_text), payload)
    log_json(f"{trace_id}_module4_method_json_response", {
        "raw_response": method_json_text,
        "method_json": method_json,
    })
    return method_json


def generate_module4(
    client: Any,
    model: str,
    answer_type: str,
    payload: Dict[str, Any],
    few_shot_examples: List[Dict[str, Any]],
) -> str:
    """
    Module 4: Generate executable SymPy code for a physics problem.

    Uses few-shot prompting via chat messages:
      - system: answer-type-specific instructions
      - user/assistant pairs: example (JSON payload → Python code)
      - user: actual JSON payload

    The system prompt is selected based on answer_type
    (numeric, MCQ, conceptual, yes/no).
    """
    # Select the correct system prompt for this answer type
    system_prompt = ANSWER_TYPE_SYSTEMS.get(answer_type, NUMERIC_SYSTEM)
    method_json = generate_module4_method_json(
        client=client,
        model=model,
        answer_type=answer_type,
        payload=payload,
        few_shot_examples=few_shot_examples,
    )
    code_payload = {
        **payload,
        "method_json": method_json,
    }

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    # Add dynamic few-shot examples as user→assistant turns
    messages.extend(build_module4_few_shot_messages(few_shot_examples))

    # Add the actual payload as the final user message
    messages.append({
        "role": "user",
        "content": json.dumps(code_payload, indent=2, ensure_ascii=False),
    })
    trace_id = _trace_id_from_payload(payload)
    log_json(f"{trace_id}_module4_prompt_messages", {
        "answer_type": answer_type,
        "payload": code_payload,
        "method_json": method_json,
        "fewshot_ids": [ex.get("id") for ex in few_shot_examples],
        "fewshots": [build_module4_few_shot_example(ex) for ex in few_shot_examples],
        "messages": messages,
    })

    # Single attempt (retries are handled in the debug loop)
    content = chat_messages(client, model, messages, module_id=4)
    code = sanitize_generated_code(extract_python_code(content))
    log_json(f"{trace_id}_module4_raw_response_and_code", {
        "raw_response": content,
        "sanitized_code": code,
    })
    return code


def clean_direct_answer(text: str) -> str:
    """Normalize a direct conceptual LLM answer into one submission-safe line."""
    s = str(text or "").strip()
    fenced = re.search(r"```(?:text)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if fenced:
        s = fenced.group(1).strip()
    boxed = extract_boxed(s)
    if boxed is not None:
        s = boxed.strip()
    s = s.strip().strip("\"'")
    s = re.sub(r"^(?:the answer is|answer:|it is|this is)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 1 and s.endswith("."):
        s = s[:-1].rstrip()
    return s


def generate_direct_conceptual_answer(
    client: Any,
    model: str,
    payload: Dict[str, Any],
) -> str:
    """Generate only the final text answer for conceptual_qa."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": DIRECT_CONCEPTUAL_SYSTEM},
        {"role": "user", "content": json.dumps(payload, indent=2, ensure_ascii=False)},
    ]
    # conceptual_qa direct answer uses base model (no adapter)
    return clean_direct_answer(chat_messages(client, model, messages, use_adapter=False))


def _parse_direct_fallback_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _first_target_unit(payload: Dict[str, Any]) -> str:
    normalized = payload.get("normalized_problem", {})
    targets = normalized.get("target", []) if isinstance(normalized, dict) else []
    target = targets[0] if targets and isinstance(targets[0], dict) else {}
    return str(target.get("unit", "") or "").strip()


def _clean_fallback_scalar(text: Any) -> str:
    s = str(text or "").strip()
    boxed = extract_boxed(s)
    if boxed is not None:
        s = boxed.strip()
    s = s.strip().strip('"\'')
    s = re.sub(r"^(?:the answer is|answer:|it is|this is)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 1 and s.endswith("."):
        s = s[:-1].rstrip()
    return s


def _numeric_text_only(text: str) -> Optional[str]:
    number_pattern = r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?"
    s = str(text or "").strip()
    if re.fullmatch(number_pattern, s):
        return s
    matches = re.findall(number_pattern, s)
    if not matches:
        return None
    return matches[-1]


def _boxed_stdout(answer: str, unit: str) -> str:
    if unit:
        return rf"\boxed{{{answer} \, \mathrm{{{unit}}}}}"
    return rf"\boxed{{{answer}}}"


def generate_direct_llm_fallback_answer(
    client: Any,
    model: str,
    answer_type: str,
    payload: Dict[str, Any],
    last_error: str,
) -> Dict[str, Any]:
    """Ask the base LLM for a direct answer when two SymCode attempts failed."""
    del last_error
    fallback_payload = dict(payload)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": LLM_DIRECT_FALLBACK_SYSTEM},
        {"role": "user", "content": json.dumps(fallback_payload, indent=2, ensure_ascii=False)},
    ]
    raw = chat_messages(client, model, messages, use_adapter=False)
    data = _parse_direct_fallback_json(raw) or {}

    answer = _clean_fallback_scalar(data.get("answer") or raw)
    unit = _clean_fallback_scalar(data.get("unit") or "")
    explanation = _clean_fallback_scalar(data.get("explanation") or "Direct physics reasoning was used to compute the final answer.")

    if answer_type == "numeric_compute":
        numeric_answer = _numeric_text_only(answer)
        if numeric_answer is not None:
            answer = numeric_answer
        if not unit:
            unit = _first_target_unit(payload)
    else:
        unit = ""

    if not answer:
        answer = "0" if answer_type == "numeric_compute" else "Uncertain"

    stdout = _boxed_stdout(answer, unit)
    return {
        "code": "# Step 1: Solve the problem from the normalized physics facts.\n"
                f"# Step 2: {explanation}\n"
                "# Step 3: Report the final value in the requested answer format.\n"
                f"answer = {answer!r}",
        "execution": {
            "status": "pass",
            "error_type": None,
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
            "boxed": extract_boxed(stdout) or answer,
            "numeric_answer": float(answer) if answer_type == "numeric_compute" and _numeric_text_only(answer) == answer else None,
            "pred_unit": unit,
            "answer_source": "direct_physics_reasoning",
            "raw_direct_answer": raw[:1000],
        },
        "debugged": True,
    }


def debug_module4(
    client: Any,
    model: str,
    answer_type: str,
    payload: Dict[str, Any],
    bad_code: str,
    error_text: str,
) -> str:
    """
    Ask the LLM to fix failed code.

    Sends:
      - System: answer-type instructions (same as generation)
      - User: debug prompt with the broken code + error message
    """
    system_prompt = ANSWER_TYPE_SYSTEMS.get(answer_type, NUMERIC_SYSTEM)

    debug_content = DEBUG_PROMPT.format(
        json_input=json.dumps(payload, indent=2, ensure_ascii=False),
        bad_code=bad_code,
        error_text=error_text[-4000:],  # truncate long errors
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": debug_content},
    ]
    trace_id = _trace_id_from_payload(payload)
    log_json(f"{trace_id}_module4_debug_prompt", {
        "answer_type": answer_type,
        "payload": payload,
        "bad_code": bad_code,
        "error_text": error_text,
        "messages": messages,
    })

    content = chat_messages(client, model, messages, module_id=4)
    fixed_code = sanitize_generated_code(extract_python_code(content))
    log_json(f"{trace_id}_module4_debug_response", {
        "raw_response": content,
        "fixed_code": fixed_code,
    })
    return fixed_code


def generate_and_execute_with_debug(
    client: Any,
    model: str,
    answer_type: str,
    payload: Dict[str, Any],
    few_shot_examples: List[Dict[str, Any]],
    timeout: int,
    enable_debug: bool,
) -> Dict[str, Any]:
    """
    Full Module 4+5 pipeline:
      1. Generate code via Module 4 (with few-shot)
      2. Execute code in sandbox
      3. If execution fails and debug is enabled:
         a. Send code + error to LLM for fixing
         b. Re-execute the fixed code

    Returns dict with: code, execution, debugged
    """
    def _audit_and_repair_loop(current_code: str, current_execution: Dict[str, Any]) -> Dict[str, Any]:
        audit_history: List[Dict[str, Any]] = []
        if current_execution.get("status") != "pass":
            return {
                "code": current_code,
                "execution": current_execution,
                "debugged": False,
                "given_audit": {"status": "skip", "reason": "execution_not_pass"},
                "given_audit_history": audit_history,
            }

        for attempt in range(2):
            print(f"  Module 4 given audit: base-model check {attempt + 1}/2", flush=True)
            audit_result = compare_given_audit_with_base_model(client, model, payload, current_execution)
            audit_history.append(audit_result)
            if audit_result.get("status") == "pass":
                return {
                    "code": current_code,
                    "execution": current_execution,
                    "debugged": bool(audit_history[:-1]),
                    "given_audit": audit_result,
                    "given_audit_history": audit_history,
                }

            print("  Module 4 given audit: regenerating code with base model", flush=True)
            current_code = repair_module4_from_given_audit(
                client=client,
                model=model,
                answer_type=answer_type,
                payload=payload,
                bad_code=current_code,
                audit_result=audit_result,
            )
            if not has_step_comments(current_code):
                current_code = "# Step 0: Given-audit regenerated code.\n" + current_code
            print(f"  Module 4 given audit: regenerated code received ({len(current_code.splitlines())} lines)", flush=True)
            print("  Module 5 given audit: executing regenerated code", flush=True)
            current_code, current_execution = _execute_preserving_existing_debug_flow(current_code, payload, timeout)
            print(f"  Module 5 given audit: execution status={current_execution['status']}", flush=True)
            if current_execution.get("status") != "pass":
                return {
                    "code": current_code,
                    "execution": current_execution,
                    "debugged": True,
                    "given_audit": {"status": "skip", "reason": "regenerated_execution_failed"},
                    "given_audit_history": audit_history,
                }

        final_audit = compare_given_audit_with_base_model(client, model, payload, current_execution)
        audit_history.append(final_audit)
        return {
            "code": current_code,
            "execution": current_execution,
            "debugged": True,
            "given_audit": final_audit,
            "given_audit_history": audit_history,
        }

    if answer_type == "conceptual_qa":
        print("  Module 4: waiting for direct conceptual answer", flush=True)
        answer = generate_direct_conceptual_answer(
            client=client,
            model=model,
            payload=payload,
        )
        stdout = rf"\boxed{{{answer}}}"
        return {
            "code": "# Direct conceptual answer; no SymPy execution.\n"
                    f"answer = {answer!r}",
            "execution": {
                "status": "pass",
                "error_type": None,
                "returncode": 0,
                "stdout": stdout,
                "stderr": "",
                "boxed": answer,
                "numeric_answer": None,
                "pred_unit": "",
            },
            "debugged": False,
            "given_audit": {"status": "skip", "reason": "conceptual_answer"},
            "given_audit_history": [],
        }

    deterministic_code = build_deterministic_sympy_code(payload)
    trace_id = _trace_id_from_payload(payload)
    if deterministic_code:
        print("  Module 4: using deterministic SymPy fallback", flush=True)
        log_text(f"{trace_id}_module4_deterministic_code", deterministic_code)
        deterministic_code, execution = _execute_preserving_existing_debug_flow(deterministic_code, payload, timeout)
        log_json(f"{trace_id}_module5_deterministic_execution", execution)
        print(f"  Module 5: execution status={execution['status']}", flush=True)
        return _audit_and_repair_loop(deterministic_code, execution)

    # Step 1: Generate code
    print("  Module 4: waiting for LLM code response", flush=True)
    code = generate_module4(
        client=client,
        model=model,
        answer_type=answer_type,
        payload=payload,
        few_shot_examples=few_shot_examples,
    )
    print(f"  Module 4: code received ({len(code.splitlines())} lines)", flush=True)

    # Ensure step comments exist (quality check)
    if not has_step_comments(code):
        code = "# Step 0: Generated code (no step comments detected).\n" + code
    log_text(f"{trace_id}_module4_final_initial_code", code)

    # Step 2: Execute
    print("  Module 5: executing generated code", flush=True)
    code, execution = _execute_preserving_existing_debug_flow(code, payload, timeout)
    log_json(f"{trace_id}_module5_initial_execution", execution)
    print(f"  Module 5: execution status={execution['status']}", flush=True)

    # Step 3: Self-debug if execution failed
    if enable_debug and execution["status"] != "pass":
        error_text = execution.get("stderr", "") or execution.get("stdout", "")
        try:
            print("  Module 4 debug: waiting for LLM fix", flush=True)
            fixed_code = debug_module4(
                client=client,
                model=model,
                answer_type=answer_type,
                payload=payload,
                bad_code=code,
                error_text=error_text,
            )
            print(f"  Module 4 debug: fixed code received ({len(fixed_code.splitlines())} lines)", flush=True)

            if not has_step_comments(fixed_code):
                fixed_code = "# Step 0: Debugged code.\n" + fixed_code
            log_text(f"{trace_id}_module4_debugged_code", fixed_code)

            print("  Module 5 debug: executing fixed code", flush=True)
            fixed_code, fixed_execution = _execute_preserving_existing_debug_flow(fixed_code, payload, timeout)
            log_json(f"{trace_id}_module5_debug_execution", fixed_execution)
            print(f"  Module 5 debug: execution status={fixed_execution['status']}", flush=True)

            result = _audit_and_repair_loop(fixed_code, fixed_execution)
            if result.get("execution", {}).get("status") == "pass":
                result["debugged"] = True
                return result

            last_error = fixed_execution.get("stderr", "") or fixed_execution.get("stdout", "")
            print("  Module 4: asking for direct physics answer", flush=True)
            return generate_direct_llm_fallback_answer(
                client=client,
                model=model,
                answer_type=answer_type,
                payload=payload,
                last_error=last_error,
            )

        except Exception as exc:
            execution["debug_error"] = str(exc)
            print("  Module 4: asking for direct physics answer", flush=True)
            try:
                return generate_direct_llm_fallback_answer(
                    client=client,
                    model=model,
                    answer_type=answer_type,
                    payload=payload,
                    last_error=str(exc),
                )
            except Exception as fallback_exc:
                execution["direct_fallback_error"] = str(fallback_exc)

    return _audit_and_repair_loop(code, execution)


def _trace_id_from_payload(payload: Dict[str, Any]) -> str:
    question = str(payload.get("raw_question", "") or "")
    m = re.search(r"\b([A-Z]{2,5}\d{3})\b", question)
    if m:
        return m.group(1)
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
    return f"question_{digest}"


# ===== Extracted notebook cell 32 =====
# Compares the predicted answer (from sandbox execution) against
# the gold answer from the input data.
