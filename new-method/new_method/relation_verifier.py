"""Fail-closed bidirectional semantic verification for the relation graph."""

from __future__ import annotations

import math
import operator
import re
from typing import Any, Dict, Mapping

from .normalizer import normalize_quantity

try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None


OPS = {"=": operator.eq, "==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
SAFE_EXPRESSION_RE = re.compile(r"^[A-Za-z0-9_+\-*/().,%\s]+$")
SAFE_FUNCTION_NAMES = {"sqrt", "sin", "cos", "tan", "exp", "log", "abs", "min", "max", "pi", "e"}
SYMBOL_RE = re.compile(r"^[A-Za-z_]\w*$")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _declared_symbols(normalized_ir: Mapping[str, Any]) -> set[str]:
    symbols = {str(item) for item in normalized_ir.get("symbols", {}).keys()}
    symbols.update(str(item.get("symbol")) for item in normalized_ir.get("givens", []) if item.get("symbol"))
    target = normalized_ir.get("target_unknown", {}).get("symbol")
    if target:
        symbols.add(str(target))
    return symbols


def validate_execution_output(execution: Mapping[str, Any], normalized_ir: Mapping[str, Any]) -> list[str]:
    """Validate the model program's public JSON contract before semantic checks."""
    runtime_errors = [str(execution[key]) for key in ("error", "execution_error", "stderr") if execution.get(key)]
    if runtime_errors:
        return [f"execution error: {item}" for item in runtime_errors]
    errors: list[str] = []
    for field in ("answer", "unit", "variables"):
        if field not in execution:
            errors.append(f"execution output missing field: {field}")
    if _number(execution.get("answer")) is None:
        errors.append("execution.answer must be a finite numeric value")
    variables = execution.get("variables")
    if not isinstance(variables, Mapping):
        errors.append("execution.variables must be an object")
    else:
        declared = _declared_symbols(normalized_ir)
        for symbol, value in variables.items():
            if not SYMBOL_RE.fullmatch(str(symbol)):
                errors.append(f"execution.variables has invalid symbol: {symbol}")
            elif str(symbol) not in declared:
                errors.append(f"execution.variables contains undeclared symbol: {symbol}")
            if _number(value) is None:
                errors.append(f"execution.variables[{symbol}] must be finite numeric")
    expected_unit = normalized_ir.get("required_output", {}).get("unit")
    actual_unit = execution.get("unit")
    if expected_unit is None:
        if actual_unit not in (None, ""):
            errors.append(f"execution.unit must be null for a dimensionless target, got {actual_unit!r}")
    elif str(actual_unit or "").strip() != str(expected_unit).strip():
        errors.append(f"execution.unit must equal requested unit {expected_unit!r}, got {actual_unit!r}")
    if _number(execution.get("answer")) is not None and actual_unit:
        quantity = normalize_quantity(execution["answer"], str(actual_unit))
        if quantity.get("status") != "ok":
            errors.append(f"execution.unit is unsupported: {actual_unit}")
    return errors


def _safe_sympify(expr: Any):
    if sp is None:
        raise RuntimeError("sympy is not installed")
    text = str(expr).strip()
    if not SAFE_EXPRESSION_RE.fullmatch(text) or "__" in text:
        raise ValueError("unsafe or unsupported expression")
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", text))
    local_dict = {name: sp.Symbol(name) for name in names - SAFE_FUNCTION_NAMES}
    local_dict.update({"sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "abs": sp.Abs, "min": sp.Min, "max": sp.Max, "pi": sp.pi, "e": sp.E})
    return sp.sympify(text, locals=local_dict)


def _evaluate(expr: Any, env: Mapping[str, Any]) -> tuple[float | None, str | None]:
    if _number(expr) is not None:
        return _number(expr), None
    try:
        parsed = _safe_sympify(expr)
        substitutions = {sp.Symbol(name): value for name, value in env.items() if _number(value) is not None}
        result = parsed.subs(substitutions)
        return (_number(result), None) if not result.free_symbols else (None, f"unresolved symbols: {sorted(map(str, result.free_symbols))}")
    except Exception as exc:
        return None, f"cannot parse expression {expr!r}: {exc}"


def _reverse_checks(lhs: Any, rhs: Any, op: str, env: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Solve an equality for every available symbol and compare the reverse result."""
    if sp is None or op not in {"=", "=="}:
        return []
    try:
        left, right = _safe_sympify(lhs), _safe_sympify(rhs)
    except Exception as exc:
        return [{"status": "unknown", "error": str(exc)}]
    symbols = sorted((left - right).free_symbols, key=str)
    checks: list[Dict[str, Any]] = []
    for symbol in symbols:
        name = str(symbol)
        candidate = _number(env.get(name))
        if candidate is None:
            continue
        known = {sp.Symbol(key): value for key, value in env.items() if key != name and _number(value) is not None}
        try:
            solutions = sp.solve(sp.Eq(left.subs(known), right.subs(known)), symbol)
            expected = [_number(item) for item in solutions]
            expected = [item for item in expected if item is not None]
            if not expected:
                checks.append({"symbol": name, "status": "unknown", "candidate": candidate, "expected": []})
                continue
            passed = any(abs(candidate - value) <= 1e-8 * max(1.0, abs(candidate), abs(value)) for value in expected)
            checks.append({"symbol": name, "status": "pass" if passed else "fail", "candidate": candidate, "expected": expected})
        except Exception as exc:
            checks.append({"symbol": name, "status": "unknown", "candidate": candidate, "expected": [], "error": str(exc)})
    return checks


def verify_bidirectional(normalized_ir: Mapping[str, Any], execution: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    execution = dict(execution or {})
    output_errors = validate_execution_output(execution, normalized_ir)
    env: Dict[str, Any] = {}
    for given in normalized_ir.get("givens", []):
        quantity = given.get("quantity", {})
        if _number(quantity.get("canonical_value")) is not None:
            env[str(given.get("symbol"))] = quantity["canonical_value"]
    variables = execution.get("variables") if isinstance(execution.get("variables"), Mapping) else {}
    for key, value in variables.items():
        if _number(value) is not None:
            env[str(key)] = float(value)
    answer = execution.get("answer")
    answer_unit = execution.get("unit")
    if _number(answer) is not None and answer_unit:
        quantity = normalize_quantity(answer, str(answer_unit))
        answer = quantity.get("canonical_value") if quantity.get("status") == "ok" else None
    target_symbol = str(normalized_ir.get("target_unknown", {}).get("symbol") or "answer")
    if _number(answer) is not None:
        env[target_symbol] = float(answer)

    relations = list(normalized_ir.get("relations", []))
    checks: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    if not relations:
        output_errors.append("IR has no verifiable relations")
    for index, relation in enumerate(relations):
        lhs, rhs = relation.get("lhs"), relation.get("rhs")
        op = relation.get("operator", "=")
        left_value, left_error = _evaluate(lhs, env)
        right_value, right_error = _evaluate(rhs, env)
        status, residual = "unknown", None
        if left_value is not None and right_value is not None:
            residual = left_value - right_value
            passed = OPS.get(op, operator.eq)(left_value, right_value) if op not in {"=", "=="} else abs(residual) <= 1e-8 * max(1.0, abs(left_value), abs(right_value))
            status = "pass" if passed else "fail"
        reverse = _reverse_checks(lhs, rhs, op, env)
        reverse_failures = [item for item in reverse if item.get("status") == "fail"]
        check = {"id": relation.get("id", f"relation_{index}"), "lhs": lhs, "rhs": rhs, "operator": op, "status": status, "residual": residual, "forward_error": left_error or right_error, "reverse": reverse}
        checks.append(check)
        if status == "fail" or reverse_failures:
            failures.append(check)
    missing_target = _number(answer) is None
    unknown_checks = [item for item in checks if item["status"] == "unknown"]
    reverse_unknown = [item for check in checks for item in check["reverse"] if item.get("status") == "unknown"]
    overall = "fail" if failures or output_errors else ("unknown" if unknown_checks or reverse_unknown or missing_target else "pass")
    feedback = [f"Output contract error: {error}" for error in output_errors]
    feedback.extend(f"Relation {item['id']} fails: {item['lhs']} {item['operator']} {item['rhs']} (residual={item['residual']})" for item in failures)
    for check in checks:
        for reverse in check["reverse"]:
            if reverse.get("status") == "fail":
                feedback.append(f"Reverse check {check['id']} fails for {reverse['symbol']}: candidate={reverse['candidate']}, expected={reverse['expected']}")
    feedback.extend(f"Relation {item['id']} cannot be evaluated: {item['forward_error'] or 'missing variable/value'}" for item in unknown_checks)
    if missing_target:
        feedback.append(f"Missing finite target answer for symbol {target_symbol}")
    return {"status": overall, "environment": env, "checks": checks, "failures": failures, "unknown": unknown_checks, "output_errors": output_errors, "feedback": feedback}
