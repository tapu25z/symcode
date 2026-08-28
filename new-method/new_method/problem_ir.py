"""Canonical intermediate representation (IR) for word problems.

The extractor is allowed to be probabilistic; every downstream stage consumes
this small, explicit contract instead of reparsing the original prose.
"""

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
    "system", "range", "identity",
}
ALLOWED_ROLES = {"constant", "variable", "measurement", "derived", "parameter"}
ALLOWED_OUTPUT_TYPES = {"number", "quantity", "ratio", "percentage", "symbolic", "tuple", "set", "interval", "matrix", "text"}
ALLOWED_PRECISIONS = {"exact", "integer", "decimal", "significant_figures"}
SYMBOL_RE = re.compile(r"^[A-Za-z_]\w*$")
SAFE_EXPRESSION_RE = re.compile(r"^[A-Za-z0-9_+\-*/().,%^\[\]\s]+$")
SAFE_CONDITION_RE = re.compile(r"^[A-Za-z0-9_+\-*/().,%^\[\]<>=!\s]+$")
ALLOWED_MATH_NAMES = {"pi", "e", "oo", "sqrt", "sin", "cos", "tan", "exp", "log", "abs", "min", "max", "int", "gcd", "lcm", "factorint", "divisors", "oct", "bin", "Tuple", "FiniteSet", "Interval", "Union", "Matrix"}
REQUIRED_RELATION_FIELDS = {"id", "kind", "lhs", "rhs", "operator", "unit", "source", "evidence", "confidence"}
REQUIRED_GIVEN_FIELDS = {"name", "symbol", "value", "unit", "role", "source"}
REQUIRED_CONDITION_FIELDS = {"kind", "expr", "source"}


def _expression_symbols(expression: Any) -> set[str]:
    return {
        name for name in re.findall(r"\b[A-Za-z_]\w*\b", str(expression or ""))
        if name not in ALLOWED_MATH_NAMES and name not in {"True", "False", "and", "or", "not"}
    }


def _validate_expression(expression: Any, path: str, declared_symbols: set[str], allow_comparison: bool = False) -> List[str]:
    text = str(expression or "").strip()
    errors: List[str] = []
    if not text:
        return [f"{path} must be a non-empty expression"]
    pattern = SAFE_CONDITION_RE if allow_comparison else SAFE_EXPRESSION_RE
    if not pattern.fullmatch(text) or "__" in text:
        errors.append(f"{path} contains unsupported characters")
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

    # Compatibility with the compact planner used by the old method.
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
            given.setdefault("value", given.get("expression"))
            given.setdefault("role", "constant")
            given.setdefault("source", str(given.get("value") or "given"))
            given.setdefault("unit", None)
    for index, relation in enumerate(out["relations"]):
        if isinstance(relation, Mapping):
            relation.setdefault("id", f"relation_{index}")
            relation.setdefault("kind", "equation")
            relation.setdefault("lhs", "")
            relation.setdefault("rhs", "")
            relation.setdefault("operator", "=")
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
    if not str(target.get("name") or "").strip():
        errors.append("target_unknown.name must be non-empty")
    target_symbol = str(target.get("symbol") or "")
    if not SYMBOL_RE.fullmatch(target_symbol):
        errors.append("target_unknown.symbol must be an ASCII Python identifier")
    required_output = ir.get("required_output", {})
    if not isinstance(required_output, Mapping):
        errors.append("required_output must be an object")
    else:
        for field in ("type", "unit", "precision", "digits", "target_count"):
            if field not in required_output:
                errors.append(f"required_output missing field: {field}")
        if required_output.get("type") not in ALLOWED_OUTPUT_TYPES:
            errors.append(f"required_output.type must be one of {sorted(ALLOWED_OUTPUT_TYPES)}")
        if required_output.get("precision") not in ALLOWED_PRECISIONS:
            errors.append(f"required_output.precision must be one of {sorted(ALLOWED_PRECISIONS)}")
        if required_output.get("target_count") != 1:
            errors.append("required_output.target_count must equal 1 for this pipeline")
        precision = required_output.get("precision")
        digits = required_output.get("digits")
        if precision in {"decimal", "significant_figures"} and (not isinstance(digits, int) or isinstance(digits, bool) or digits < 0):
            errors.append("required_output.digits must be a non-negative integer for decimal/significant_figures precision")
        if target.get("unit") and required_output.get("unit") != target.get("unit"):
            errors.append("required_output.unit must match target_unknown.unit")
    declared_symbols = {target_symbol} if SYMBOL_RE.fullmatch(target_symbol) else set()
    seen_given_symbols: set[str] = set()
    for index, given in enumerate(ir.get("givens", [])):
        if not isinstance(given, Mapping):
            errors.append(f"given[{index}] must be an object")
            continue
        missing = sorted(REQUIRED_GIVEN_FIELDS - set(given))
        if missing:
            errors.append(f"given[{index}] missing fields: {missing}")
        symbol = str(given.get("symbol") or "")
        if not SYMBOL_RE.fullmatch(symbol):
            errors.append(f"given[{index}].symbol must be an ASCII Python identifier")
        elif symbol in seen_given_symbols or symbol == target_symbol:
            errors.append(f"given[{index}].symbol is duplicated: {symbol}")
        else:
            seen_given_symbols.add(symbol)
            declared_symbols.add(symbol)
        if given.get("role") not in ALLOWED_ROLES:
            errors.append(f"given[{index}].role must be one of {sorted(ALLOWED_ROLES)}")
        if not str(given.get("name") or "").strip():
            errors.append(f"given[{index}].name must be non-empty")
        if not str(given.get("source") or "").strip():
            errors.append(f"given[{index}].source must be non-empty")
    relations = ir.get("relations", [])
    if not relations and required_output.get("type") in ALLOWED_OUTPUT_TYPES:
        errors.append("at least one relation is required for a numeric target")
    # A definition may introduce one intermediate unknown on its bare-symbol lhs.
    intermediate_definitions: dict[str, int] = {}
    initially_declared = set(declared_symbols)
    for relation in relations:
        if isinstance(relation, Mapping) and relation.get("kind") == "definition":
            lhs = str(relation.get("lhs") or "").strip()
            if SYMBOL_RE.fullmatch(lhs) and lhs not in initially_declared:
                declared_symbols.add(lhs)
                intermediate_definitions[lhs] = intermediate_definitions.get(lhs, 0) + 1
    for symbol, count in intermediate_definitions.items():
        if count > 1:
            errors.append(f"intermediate symbol has multiple definitions: {symbol}")
    seen_relation_ids: set[str] = set()
    for index, relation in enumerate(relations):
        if not isinstance(relation, Mapping):
            errors.append(f"relation[{index}] must be an object")
            continue
        missing = sorted(REQUIRED_RELATION_FIELDS - set(relation))
        if missing:
            errors.append(f"relation[{index}] missing fields: {missing}")
        relation_id = str(relation.get("id") or "")
        if not relation_id:
            errors.append(f"relation[{index}].id must be non-empty")
        elif relation_id in seen_relation_ids:
            errors.append(f"relation[{index}].id is duplicated: {relation_id}")
        else:
            seen_relation_ids.add(relation_id)
        errors.extend(_validate_expression(relation.get("lhs"), f"relation[{index}].lhs", declared_symbols))
        errors.extend(_validate_expression(relation.get("rhs"), f"relation[{index}].rhs", declared_symbols))
        if relation.get("operator") not in {"=", "==", "<", "<=", ">", ">=", "!="}:
            errors.append(f"relation[{index}] has unsupported operator")
        if relation.get("kind") not in ALLOWED_RELATION_KINDS:
            errors.append(f"relation[{index}].kind must be one of {sorted(ALLOWED_RELATION_KINDS)}")
        if not str(relation.get("source") or "").strip():
            errors.append(f"relation[{index}].source must be non-empty")
        if not str(relation.get("evidence") or "").strip():
            errors.append(f"relation[{index}].evidence must be non-empty")
        confidence = relation.get("confidence")
        try:
            if not 0 <= float(confidence) <= 1:
                errors.append(f"relation[{index}].confidence must be in [0, 1]")
        except (TypeError, ValueError):
            errors.append(f"relation[{index}].confidence must be numeric")
    for index, condition in enumerate(ir.get("conditions", [])):
        if not isinstance(condition, Mapping):
            errors.append(f"condition[{index}] must be an object")
            continue
        missing = sorted(REQUIRED_CONDITION_FIELDS - set(condition))
        if missing:
            errors.append(f"condition[{index}] missing fields: {missing}")
        if not str(condition.get("kind") or "").strip():
            errors.append(f"condition[{index}].kind must be non-empty")
        if not str(condition.get("source") or "").strip():
            errors.append(f"condition[{index}].source must be non-empty")
        errors.extend(_validate_expression(condition.get("expr"), f"condition[{index}].expr", declared_symbols, allow_comparison=True))
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
