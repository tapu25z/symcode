"""Fail-closed numeric and symbolic verification for the relation graph."""

from __future__ import annotations

import math
import operator
import re
from typing import Any, Dict, Mapping

try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None


OPS = {"!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
NUMERIC_OUTPUT_TYPES = {"number", "quantity", "ratio", "percentage"}
STRUCTURED_OUTPUT_TYPES = {"symbolic", "tuple", "set", "interval", "matrix", "text"}
SAFE_EXPRESSION_RE = re.compile(r"^[A-Za-z0-9_+\-*/().,%\[\]\s]+$")
SAFE_CONDITION_RE = re.compile(r"^[A-Za-z0-9_+\-*/().,%\[\]<>=!\s]+$")
SAFE_FUNCTION_NAMES = {"sqrt", "sin", "cos", "tan", "exp", "log", "abs", "min", "max", "int", "gcd", "lcm", "factorint", "divisors", "oct", "bin", "pi", "e", "I", "oo", "Tuple", "FiniteSet", "Interval", "Union", "Matrix"}
SYMBOL_RE = re.compile(r"^[A-Za-z_]\w*$")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_sympify(value: Any, allow_condition: bool = False):
    if sp is None:
        raise RuntimeError("sympy is not installed")
    if isinstance(value, sp.Basic):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError("non-finite numeric value")
        return sp.sympify(value)
    text = str(value).strip()
    text = text.replace("π", "pi").replace("∞", "oo").replace("²", "**2").replace("³", "**3")
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = text.replace("\\pi", "pi").replace("\\infty", "oo")
    text = re.sub(r"(?<=\d)\s+(?=(sqrt|sin|cos|tan|log|exp)\s*\()", "*", text)
    text = re.sub(r"(?<=\d)\s*pi\b", "*pi", text)
    text = re.sub(r"(?<=[0-9)\]])\s*i\b", "*I", text)
    text = re.sub(r"(?<=[0-9)\]])(?=I\b)", "*", text)
    text = re.sub(r"\bi\b", "I", text)
    pattern = SAFE_CONDITION_RE if allow_condition else SAFE_EXPRESSION_RE
    if not text or not pattern.fullmatch(text) or "__" in text:
        raise ValueError("unsafe or unsupported expression")
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", text))
    local_dict = {name: sp.Symbol(name) for name in names - SAFE_FUNCTION_NAMES - {"Ropen", "Lopen", "open"}}
    local_dict.update({
        "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "abs": sp.Abs, "min": sp.Min,
        "max": sp.Max, "pi": sp.pi, "e": sp.E, "I": sp.I, "oo": sp.oo, "Tuple": sp.Tuple,
        "FiniteSet": sp.FiniteSet, "Interval": sp.Interval, "Union": sp.Union,
        "Matrix": sp.Matrix, "gcd": sp.gcd, "lcm": sp.ilcm,
        "factorint": sp.factorint, "divisors": sp.divisors, "oct": oct, "bin": bin,
    })
    parsed = sp.sympify(text, locals=local_dict)
    if isinstance(parsed, tuple):
        parsed = sp.Tuple(*parsed)
    return parsed


def _equivalent(left: Any, right: Any) -> bool:
    try:
        if left == right:
            return True
        if isinstance(left, sp.MatrixBase) and isinstance(right, sp.MatrixBase) and left.shape == right.shape:
            return all(_equivalent(a, b) for a, b in zip(list(left), list(right)))
        if isinstance(left, (tuple, list, sp.Tuple)) and isinstance(right, (tuple, list, sp.Tuple)) and len(left) == len(right):
            return all(_equivalent(a, b) for a, b in zip(left, right))
        if isinstance(left, sp.Set) and isinstance(right, sp.Set):
            return left.symmetric_difference(right) == sp.EmptySet
        left_number, right_number = _number(left.evalf() if hasattr(left, "evalf") else left), _number(right.evalf() if hasattr(right, "evalf") else right)
        if left_number is not None and right_number is not None:
            scale = max(1.0, abs(left_number), abs(right_number))
            return abs(left_number - right_number) <= 1e-8 * scale
        difference = sp.simplify(left - right)
        return difference == 0 or bool(getattr(difference, "equals", lambda _: False)(0))
    except Exception:
        try:
            return bool(left.equals(right))
        except Exception:
            return False


def _declared_symbols(normalized_ir: Mapping[str, Any]) -> set[str]:
    symbols = {str(item) for item in normalized_ir.get("symbols", {}).keys()}
    target = normalized_ir.get("target_unknown", {}).get("symbol")
    if target:
        symbols.add(str(target))
    return symbols


def _is_valid_canonical(value: Any, output_type: str) -> bool:
    try:
        parsed = _safe_sympify(value)
        if output_type in NUMERIC_OUTPUT_TYPES:
            return not parsed.free_symbols and bool(parsed.is_number) and _number(parsed.evalf()) is not None
        return output_type in STRUCTURED_OUTPUT_TYPES
    except Exception:
        return False


def validate_execution_output(execution: Mapping[str, Any], normalized_ir: Mapping[str, Any]) -> list[str]:
    runtime_errors = [str(execution[key]) for key in ("error", "execution_error", "stderr") if execution.get(key)]
    if runtime_errors:
        return [f"execution error: {item}" for item in runtime_errors]
    errors: list[str] = []
    required_fields = ("answer", "canonical_answer", "answer_type", "unit", "variables")
    for field in required_fields:
        if field not in execution:
            errors.append(f"execution output missing field: {field}")
    expected_type = str(normalized_ir.get("required_output", {}).get("type") or "")
    if execution.get("answer_type") != expected_type:
        errors.append(f"execution.answer_type must equal {expected_type!r}, got {execution.get('answer_type')!r}")
    answer = execution.get("answer")
    if expected_type in NUMERIC_OUTPUT_TYPES:
        if not _is_valid_canonical(execution.get("canonical_answer"), expected_type):
            errors.append("execution.canonical_answer must be a finite numeric expression")
        if not isinstance(answer, (int, float, str)) or isinstance(answer, bool) or not str(answer).strip():
            errors.append("execution.answer must be a non-empty numeric display value")
    elif expected_type in STRUCTURED_OUTPUT_TYPES:
        if not isinstance(answer, str) or not answer.strip():
            errors.append("execution.answer must be a non-empty display string")
        if not isinstance(execution.get("canonical_answer"), str) or not _is_valid_canonical(execution.get("canonical_answer"), expected_type):
            errors.append("execution.canonical_answer must be a valid symbolic/structured expression")
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
            try:
                _safe_sympify(value)
            except Exception:
                errors.append(f"execution.variables[{symbol}] must be finite numeric or a safe symbolic expression")
    expected_unit = normalized_ir.get("required_output", {}).get("unit")
    actual_unit = execution.get("unit")
    if expected_unit is None:
        if actual_unit not in (None, ""):
            errors.append(f"execution.unit must be null for a unitless target, got {actual_unit!r}")
    elif str(actual_unit or "").strip() != str(expected_unit).strip():
        errors.append(f"execution.unit must equal requested unit {expected_unit!r}, got {actual_unit!r}")
    return errors


def _substitutions(env: Mapping[str, Any], exclude: str | None = None) -> dict[Any, Any]:
    result = {}
    for name, value in env.items():
        if name == exclude:
            continue
        try:
            result[sp.Symbol(name)] = _safe_sympify(value)
        except Exception:
            continue
    return result


def _evaluate(expr: Any, env: Mapping[str, Any]) -> tuple[Any | None, str | None]:
    try:
        return _safe_sympify(expr).subs(_substitutions(env)), None
    except Exception as exc:
        return None, f"cannot parse expression {expr!r}: {exc}"


def _check_condition(condition: Mapping[str, Any], env: Mapping[str, Any]) -> tuple[str, str | None]:
    expr = str(condition.get("expr") or "").strip()
    kind = str(condition.get("kind") or "").strip().lower()
    if kind == "integer":
        match = re.fullmatch(r"([A-Za-z_]\w*)", expr) or re.fullmatch(r"([A-Za-z_]\w*)\s*==\s*int\s*\(\s*\1\s*\)", expr)
        if match:
            name = match.group(1)
            if name not in env:
                return "unknown", f"missing integer variable: {name}"
            value = _safe_sympify(env[name])
            if value.is_integer is True:
                return "pass", None
            if value.is_number and _number(value.evalf()) is not None:
                return ("pass", None) if float(value.evalf()).is_integer() else ("fail", None)
            return "unknown", f"unresolved integer value: {value}"
    evaluated = _safe_sympify(expr, allow_condition=True).subs(_substitutions(env))
    if evaluated in (True, sp.true):
        return "pass", None
    if evaluated in (False, sp.false):
        return "fail", None
    return "unknown", f"unresolved condition: {evaluated}"


def _reverse_checks(lhs: Any, rhs: Any, op: str, env: Mapping[str, Any]) -> list[Dict[str, Any]]:
    if sp is None or op not in {"=", "=="}:
        return []
    try:
        left, right = _safe_sympify(lhs), _safe_sympify(rhs)
    except Exception as exc:
        return [{"status": "unknown", "error": str(exc)}]
    try:
        relation_symbols = (left - right).free_symbols
    except Exception:
        return []
    checks: list[Dict[str, Any]] = []
    for symbol in sorted(relation_symbols, key=str):
        name = str(symbol)
        if name not in env:
            continue
        candidate = env[name]
        try:
            candidate = _safe_sympify(env[name])
            if candidate == symbol:
                # A free parameter is an input symbol, not a computed value to audit in reverse.
                continue
            solutions = sp.solve(sp.Eq(left.subs(_substitutions(env, name)), right.subs(_substitutions(env, name))), symbol)
            if not solutions:
                checks.append({"symbol": name, "status": "unknown", "candidate": str(candidate), "expected": []})
                continue
            passed = any(_equivalent(candidate, expected) for expected in solutions)
            checks.append({"symbol": name, "status": "pass" if passed else "fail", "candidate": str(candidate), "expected": [str(item) for item in solutions]})
        except Exception as exc:
            checks.append({"symbol": name, "status": "unknown", "candidate": str(candidate), "expected": [], "error": str(exc)})
    return checks


def verify_bidirectional(normalized_ir: Mapping[str, Any], execution: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    execution = dict(execution or {})
    output_errors = validate_execution_output(execution, normalized_ir)
    env: Dict[str, Any] = {}
    for given in normalized_ir.get("givens", []):
        value = given.get("quantity", {}).get("canonical_value")
        if value is not None:
            env[str(given.get("symbol"))] = value
    variables = execution.get("variables") if isinstance(execution.get("variables"), Mapping) else {}
    for key, value in variables.items():
        env.setdefault(str(key), value)
    canonical_answer = execution.get("canonical_answer")
    target_symbol = str(normalized_ir.get("target_unknown", {}).get("symbol") or "answer")
    if _is_valid_canonical(canonical_answer, str(normalized_ir.get("required_output", {}).get("type") or "")):
        env[target_symbol] = canonical_answer

    relations = list(normalized_ir.get("relations", []))
    checks: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    if not relations:
        output_errors.append("IR has no verifiable relations")
    for index, relation in enumerate(relations):
        lhs, rhs, op = relation.get("lhs"), relation.get("rhs"), relation.get("operator", "=")
        left_value, left_error = _evaluate(lhs, env)
        right_value, right_error = _evaluate(rhs, env)
        status, residual = "unknown", None
        if left_value is not None and right_value is not None:
            if op in {"=", "=="}:
                passed = _equivalent(left_value, right_value)
                try:
                    residual = str(sp.simplify(left_value - right_value))
                except Exception:
                    residual = None
            else:
                left_number, right_number = _number(left_value), _number(right_value)
                passed = left_number is not None and right_number is not None and OPS[op](left_number, right_number)
                residual = None if left_number is None or right_number is None else left_number - right_number
            status = "pass" if passed else "fail"
        reverse = _reverse_checks(lhs, rhs, op, env)
        check = {"id": relation.get("id", f"relation_{index}"), "lhs": lhs, "rhs": rhs, "operator": op, "status": status, "residual": residual, "forward_error": left_error or right_error, "reverse": reverse}
        checks.append(check)
        if status == "fail" or any(item.get("status") == "fail" for item in reverse):
            failures.append(check)
    condition_checks: list[Dict[str, Any]] = []
    condition_failures: list[Dict[str, Any]] = []
    for index, condition in enumerate(normalized_ir.get("conditions", [])):
        expr = condition.get("expr")
        try:
            status, error = _check_condition(condition, env)
        except Exception as exc:
            status, error = "unknown", str(exc)
        check = {"id": f"condition_{index}", "kind": condition.get("kind"), "expr": expr, "status": status, "error": error}
        condition_checks.append(check)
        if status == "fail":
            condition_failures.append(check)
    missing_target = target_symbol not in env
    unknown_checks = [item for item in checks if item["status"] == "unknown"]
    unknown_conditions = [item for item in condition_checks if item["status"] == "unknown"]
    overall = "fail" if failures or condition_failures or output_errors else ("unknown" if unknown_checks or unknown_conditions or missing_target else "pass")
    feedback = [f"Output contract error: {error}" for error in output_errors]
    feedback.extend(f"Relation {item['id']} fails: {item['lhs']} {item['operator']} {item['rhs']} (residual={item['residual']})" for item in failures)
    for check in checks:
        for reverse in check["reverse"]:
            if reverse.get("status") == "fail":
                feedback.append(f"Reverse check {check['id']} fails for {reverse['symbol']}: candidate={reverse['candidate']}, expected={reverse['expected']}")
    feedback.extend(f"Relation {item['id']} cannot be evaluated: {item['forward_error'] or 'missing variable/value'}" for item in unknown_checks)
    feedback.extend(f"Condition {item['expr']} fails" for item in condition_failures)
    feedback.extend(f"Condition {item['expr']} cannot be evaluated: {item['error']}" for item in unknown_conditions)
    if missing_target:
        feedback.append(f"Missing canonical target answer for symbol {target_symbol}")
    return {"status": overall, "environment": {key: str(value) for key, value in env.items()}, "checks": checks, "condition_checks": condition_checks, "failures": failures, "unknown": unknown_checks, "output_errors": output_errors, "feedback": feedback}
