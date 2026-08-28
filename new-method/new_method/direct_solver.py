"""Deterministic execution for simple verified IR graphs."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None


SAFE_FUNCTIONS = {
    "sqrt", "sin", "cos", "tan", "exp", "log", "Abs", "abs", "Min", "Max",
    "min", "max", "gcd", "lcm", "factorint", "divisors", "totient",
    "isprime", "nextprime", "prime", "comb", "perm", "factorial", "Mod",
    "Tuple", "FiniteSet", "Interval", "Union", "Intersection", "Matrix",
    "pi", "e", "I", "oo",
}
ASSIGNABLE_KINDS = {
    "definition", "sequential_step", "balance", "rate", "partition",
    "percentage", "conversion", "calculation", "step", "sequence",
    "accumulation", "combinatorics", "geometry", "identity",
}
SYMBOL_RE = re.compile(r"^[A-Za-z_]\w*$")


def _locals(env: Mapping[str, Any]) -> dict[str, Any]:
    if sp is None:
        return {}
    local = {name: sp.Symbol(name) for name in re.findall(r"\b[A-Za-z_]\w*\b", " ".join(env.keys()))}
    local.update({
        "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "Abs": sp.Abs, "abs": sp.Abs,
        "Min": sp.Min, "Max": sp.Max, "min": sp.Min, "max": sp.Max,
        "gcd": sp.gcd, "lcm": sp.ilcm, "factorint": sp.factorint,
        "divisors": sp.divisors, "totient": sp.totient, "isprime": sp.isprime,
        "nextprime": sp.nextprime, "prime": sp.prime, "comb": math.comb,
        "perm": math.perm, "factorial": math.factorial, "Mod": sp.Mod,
        "Tuple": sp.Tuple, "FiniteSet": sp.FiniteSet, "Interval": sp.Interval,
        "Union": sp.Union, "Intersection": sp.Intersection, "Matrix": sp.Matrix,
        "pi": sp.pi, "e": sp.E, "I": sp.I, "oo": sp.oo,
    })
    for key, value in env.items():
        if SYMBOL_RE.fullmatch(str(key)):
            local[str(key)] = value
    return local


def _parse(value: Any, env: Mapping[str, Any]):
    if sp is None:
        raise RuntimeError("sympy is not installed")
    if isinstance(value, sp.Basic):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError("non-finite numeric value")
        return sp.sympify(value)
    if isinstance(value, (list, tuple)):
        return sp.Tuple(*[_parse(item, env) for item in value])
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError("empty expression")
    text = text.replace("^", "**").replace("π", "pi").replace("∞", "oo")
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", text))
    local = _locals({**{name: sp.Symbol(name) for name in names - SAFE_FUNCTIONS}, **env})
    return sp.sympify(text, locals=local)


def _eval_expr(expr: Any, env: Mapping[str, Any]):
    subs_call = re.fullmatch(r"\s*([A-Za-z_]\w*)\.subs\(\s*([A-Za-z_]\w*)\s*,\s*(.+?)\s*\)\s*", str(expr))
    if subs_call:
        base_name, symbol_name, value_text = subs_call.groups()
        base = _parse(env.get(base_name, base_name), env)
        value = _parse(value_text, env).subs({sp.Symbol(key): val for key, val in env.items() if SYMBOL_RE.fullmatch(key)})
        return sp.simplify(base.subs(sp.Symbol(symbol_name), value))
    parsed = _parse(expr, env)
    return sp.simplify(parsed.subs({sp.Symbol(key): val for key, val in env.items() if SYMBOL_RE.fullmatch(key)}))


def _json_value(value: Any) -> Any:
    if sp is not None:
        if isinstance(value, sp.Integer):
            return int(value)
        if isinstance(value, sp.Rational) and not isinstance(value, sp.Integer):
            return str(value)
        if isinstance(value, sp.Float):
            number = float(value)
            return int(number) if number.is_integer() else number
        if isinstance(value, sp.Tuple):
            return [_json_value(item) for item in value]
        if isinstance(value, sp.Set):
            return [_json_value(item) for item in sorted(value, key=str)]
        if isinstance(value, sp.Symbol):
            return str(value)
        if isinstance(value, sp.Basic):
            return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def try_direct_solve(normalized_ir: Mapping[str, Any]) -> dict[str, Any] | None:
    """Solve acyclic definition-style IR without asking the code generator.

    This intentionally handles only transparent computations. Constraints,
    inequalities and unresolved sums still go through the existing LLM codegen
    path.
    """
    if sp is None:
        return None
    env: dict[str, Any] = {}
    for given in normalized_ir.get("givens", []):
        symbol = str(given.get("symbol") or "")
        quantity = given.get("quantity", {}) if isinstance(given, Mapping) else {}
        value = quantity.get("canonical_value")
        if SYMBOL_RE.fullmatch(symbol) and value not in (None, ""):
            try:
                env[symbol] = _parse(value, env)
            except Exception:
                env[symbol] = sp.Symbol(str(value))

    pending = [
        relation for relation in normalized_ir.get("relations", [])
        if isinstance(relation, Mapping)
        and relation.get("operator", "=") in {"=", "=="}
        and str(relation.get("kind", "definition")) in ASSIGNABLE_KINDS
        and SYMBOL_RE.fullmatch(str(relation.get("lhs") or ""))
    ]
    for _ in range(len(pending) + 1):
        progressed = False
        remaining = []
        for relation in pending:
            lhs = str(relation.get("lhs"))
            try:
                env[lhs] = _eval_expr(relation.get("rhs"), env)
                progressed = True
            except Exception:
                remaining.append(relation)
        pending = remaining
        if not progressed:
            break

    target = str(normalized_ir.get("target_unknown", {}).get("symbol") or "answer")
    if target not in env:
        return None
    answer = _json_value(env[target])
    output_type = str(normalized_ir.get("required_output", {}).get("type") or "number")
    unit = normalized_ir.get("required_output", {}).get("unit")
    variables = {key: _json_value(value) for key, value in env.items()}
    return {
        "answer": answer,
        "canonical_answer": answer,
        "answer_type": output_type,
        "unit": unit,
        "variables": variables,
        "_sandbox_status": "direct",
        "_stdout": "",
        "_traceback": None,
    }
