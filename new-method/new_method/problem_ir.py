"""Universal intermediate representation (IR) for mathematical word problems."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, TypedDict


class ProblemIR(TypedDict, total=False):
    target_unknown: Dict[str, Any]
    givens: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    conditions: List[Any]
    required_output: Dict[str, Any]
    extraction_notes: List[str]


REQUIRED_KEYS = ("target_unknown", "givens", "relations", "conditions", "required_output")
ALLOWED_RELATION_KINDS = {
    "equation", "inequality", "definition", "conservation", "proportion", "ordering",
    "system", "range", "identity", "congruence", "combinatorics", "geometry",
    "sequence", "calculus", "property", "general", "mod", "count",
    "sequential_step", "accumulation", "balance", "rate", "partition", "percentage", "conversion", "step", "calculation"
}
ALLOWED_ROLES = {"constant", "variable", "measurement", "derived", "parameter", "point", "function", "sequence", "set", "entity"}
ALLOWED_OUTPUT_TYPES = {
    "number", "quantity", "ratio", "percentage", "symbolic", "tuple", "set",
    "interval", "matrix", "text", "list", "expression", "root_set", "complex", "exact", "general"
}
ALLOWED_PRECISIONS = {"exact", "integer", "decimal", "significant_figures"}
SYMBOL_RE = re.compile(r"^[A-Za-z_]\w*$")
SAFE_EXPRESSION_RE = re.compile(r"^[A-Za-z0-9_+\-*/().,%^\[\]<>=!~&|:\s\\\{\}\$]+$")
SAFE_CONDITION_RE = re.compile(r"^[A-Za-z0-9_+\-*/().,%^\[\]<>=!~&|:\s\\\{\}\$]+$")

ALLOWED_MATH_NAMES = {
    "pi", "e", "oo", "I", "sqrt", "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "exp", "log", "ln", "abs", "Abs", "min", "max", "Min", "Max",
    "int", "float", "gcd", "lcm", "igcd", "ilcm", "factorint", "divisors", "divisor_count",
    "totient", "isprime", "nextprime", "prime", "oct", "bin", "hex",
    "Tuple", "FiniteSet", "Interval", "Union", "Intersection", "Matrix", "Set",
    "comb", "perm", "factorial", "Mod", "mod", "pmod", "Sum", "sum", "Product",
    "Point", "Line", "Segment", "Ray", "Circle", "Triangle", "Polygon",
    "diff", "integrate", "limit", "series", "expand", "factor", "simplify",
    "conjugate", "re", "im", "arg"
}
REQUIRED_RELATION_FIELDS = {"id", "kind", "lhs", "rhs", "operator", "unit", "source", "evidence", "confidence"}
REQUIRED_GIVEN_FIELDS = {"name", "symbol", "value", "unit", "role", "source"}
REQUIRED_CONDITION_FIELDS = {"kind", "expr", "source"}


def _expression_symbols(expression: Any) -> set[str]:
    return {
        name for name in re.findall(r"\b[A-Za-z_]\w*\b", str(expression or ""))
        if name not in ALLOWED_MATH_NAMES and name not in {"True", "False", "and", "or", "not", "in", "is"}
    }


def _validate_expression(expression: Any, path: str, declared_symbols: set[str], allow_comparison: bool = False) -> List[str]:
    text = str(expression or "").strip()
    errors: List[str] = []
    if not text:
        return [f"{path} must be a non-empty expression"]
    if "__" in text:
        errors.append(f"{path} contains unsupported dunder characters")
    unknown = sorted(_expression_symbols(text) - declared_symbols)
    if unknown:
        errors.append(f"{path} references undeclared symbols: {unknown}")
    return errors


def empty_ir() -> ProblemIR:
    return {
        "target_unknown": {"name": "answer", "symbol": "x", "unit": None, "dimension": None},
        "givens": [],
        "relations": [],
        "conditions": [],
        "required_output": {"type": "number", "unit": None, "precision": "exact", "digits": None, "target_count": 1},
        "extraction_notes": [],
    }


def normalize_ir_shape(raw: Mapping[str, Any] | None) -> ProblemIR:
    """Coerce imperfect LLM JSON into the canonical list/dict shape."""
    out = empty_ir()
    if not isinstance(raw, Mapping):
        return out
    for key in REQUIRED_KEYS:
        value = raw.get(key)
        if key in ("givens", "relations", "conditions"):
            if isinstance(value, list):
                out[key] = value
            elif value is not None:
                out[key] = [value]
        elif isinstance(value, Mapping):
            out[key] = dict(value)
    if isinstance(raw.get("extraction_notes"), list):
        out["extraction_notes"] = list(raw["extraction_notes"])

    # Compatibility with alternative key aliases
    if not out["givens"] and isinstance(raw.get("given"), list):
        out["givens"] = raw["given"]
    if not out["conditions"] and isinstance(raw.get("notes"), list):
        out["conditions"] = raw["notes"]
    if out["target_unknown"].get("name") == "answer" and raw.get("target"):
        out["target_unknown"]["name"] = str(raw["target"])
    out["target_unknown"].setdefault("unit", None)
    out["target_unknown"].setdefault("dimension", None)
    out["required_output"].setdefault("unit", out["target_unknown"].get("unit"))
    out["required_output"].setdefault("precision", "exact")
    out["required_output"].setdefault("digits", None)
    out["required_output"].setdefault("target_count", 1)

    for index, given in enumerate(out["givens"]):
        if isinstance(given, Mapping):
            given.setdefault("name", given.get("symbol") or f"given_{index}")
            given.setdefault("symbol", given.get("name") or f"given_{index}")
            given.setdefault("value", given.get("expression", given.get("val")))
            given.setdefault("role", "constant")
            given.setdefault("source", str(given.get("value") or "given"))
            given.setdefault("unit", None)

    for index, relation in enumerate(out["relations"]):
        if isinstance(relation, Mapping):
            relation.setdefault("id", f"relation_{index}")
            relation.setdefault("kind", "equation")
            relation.setdefault("lhs", relation.get("left", ""))
            relation.setdefault("rhs", relation.get("right", ""))
            relation.setdefault("operator", relation.get("op", "="))
            relation.setdefault("unit", None)
            relation.setdefault("source", "extracted relation")
            relation.setdefault("evidence", relation.get("source") or "extracted relation")
            relation.setdefault("confidence", 1.0)

    for condition in out["conditions"]:
        if isinstance(condition, Mapping):
            condition.setdefault("kind", "assumption")
            condition.setdefault("expr", "")
            condition.setdefault("source", str(condition.get("expr") or "condition"))
    return out


def validate_ir(ir: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in REQUIRED_KEYS:
        if key not in ir:
            errors.append(f"missing top-level key: {key}")
    target = ir.get("target_unknown", {})
    if not isinstance(target, Mapping):
        errors.append("target_unknown must be an object")
        target = {}
    for field in ("name", "symbol", "unit", "dimension"):
        if field not in target:
            errors.append(f"target_unknown missing field: {field}")
    target_symbol = str(target.get("symbol") or target.get("name") or "")
    if not str(target_symbol).strip():
        errors.append("target_unknown.symbol must be non-empty")
    required_output = ir.get("required_output", {})
    if not isinstance(required_output, Mapping):
        errors.append("required_output must be an object")

    declared_symbols = {target_symbol} if SYMBOL_RE.fullmatch(target_symbol) else set()
    for index, given in enumerate(ir.get("givens", [])):
        if not isinstance(given, Mapping):
            continue
        missing = sorted(REQUIRED_GIVEN_FIELDS - set(given))
        if missing:
            errors.append(f"given[{index}] missing fields: {missing}")
        symbol = str(given.get("symbol") or "")
        if symbol:
            declared_symbols.add(symbol)

    relations = ir.get("relations", [])
    if not relations and required_output.get("type") in ALLOWED_OUTPUT_TYPES:
        errors.append("at least one relation is required for a numeric target")

    for relation in relations:
        if isinstance(relation, Mapping) and relation.get("kind") == "definition":
            lhs = str(relation.get("lhs") or "").strip()
            if SYMBOL_RE.fullmatch(lhs):
                declared_symbols.add(lhs)

    for index, relation in enumerate(relations):
        if not isinstance(relation, Mapping):
            continue
        missing = sorted(REQUIRED_RELATION_FIELDS - set(relation))
        if missing:
            errors.append(f"relation[{index}] missing fields: {missing}")
        errors.extend(_validate_expression(relation.get("lhs"), f"relation[{index}].lhs", declared_symbols))
        errors.extend(_validate_expression(relation.get("rhs"), f"relation[{index}].rhs", declared_symbols))

    return errors


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse JSON from a model response, including fenced or surrounding prose."""
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)
    try:
        value = json.loads(candidate)
        return dict(value) if isinstance(value, Mapping) else {}
    except json.JSONDecodeError:
        starts = [m.start() for m in re.finditer(r"\{", candidate)]
        for start in starts:
            depth = 0
            in_string = False
            escaped = False
            for pos in range(start, len(candidate)):
                char = candidate[pos]
                if in_string and escaped:
                    escaped = False
                elif in_string and char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = not in_string
                elif not in_string and char == "{":
                    depth += 1
                elif not in_string and char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            value = json.loads(candidate[start : pos + 1])
                            return dict(value) if isinstance(value, Mapping) else {}
                        except json.JSONDecodeError:
                            break
    return {}
