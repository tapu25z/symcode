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

ACG_EDGE_KINDS = {
    "definition", "constraint", "transformation", "aggregation",
    "selection", "verification", "annotation",
}
ACG_NODE_TYPES = {
    "quantity", "variable", "object", "set", "sequence", "function",
    "point", "expression", "boolean", "unknown",
}


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


def empty_acg_ir() -> Dict[str, Any]:
    """Return the tolerant, dataset-agnostic Adaptive Computation Graph shape."""
    return {
        "version": "acg-ir-v1",
        "problem_metadata": {"dataset": None, "source_id": None, "domain_hints": [], "language": "en"},
        "target": {
            "id": "answer", "name": "requested answer", "symbol": "answer",
            "unit": None, "dimension": None, "output_type": "number",
            "precision": "exact", "target_count": 1,
        },
        "nodes": [],
        "edges": [],
        "conditions": [],
        "solver_hints": [],
        "extraction_notes": [],
    }


def normalize_acg_shape(raw: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Coerce model output into ACG-IR v1 without imposing a problem taxonomy."""
    out = empty_acg_ir()
    if not isinstance(raw, Mapping):
        return out
    out["version"] = str(raw.get("version") or "acg-ir-v1")
    if isinstance(raw.get("problem_metadata"), Mapping):
        out["problem_metadata"] = {**out["problem_metadata"], **dict(raw["problem_metadata"])}
    target = raw.get("target")
    if not isinstance(target, Mapping):
        # Permit a small compatibility alias so an imperfect ACG response is recoverable.
        target = raw.get("target_unknown") if isinstance(raw.get("target_unknown"), Mapping) else {}
    out["target"] = {**out["target"], **dict(target)}
    if out["target"].get("symbol") in (None, ""):
        out["target"]["symbol"] = str(out["target"].get("id") or out["target"].get("name") or "answer")
    out["target"].setdefault("id", out["target"].get("symbol") or "answer")
    out["target"].setdefault("name", out["target"].get("symbol") or "answer")
    out["target"].setdefault("unit", None)
    out["target"].setdefault("dimension", None)
    out["target"].setdefault("output_type", "number")
    out["target"].setdefault("precision", "exact")
    out["target"].setdefault("target_count", 1)

    nodes = raw.get("nodes", [])
    out["nodes"] = list(nodes) if isinstance(nodes, list) else [nodes]
    edges = raw.get("edges", raw.get("relations", []))
    out["edges"] = list(edges) if isinstance(edges, list) else [edges]
    conditions = raw.get("conditions", [])
    out["conditions"] = list(conditions) if isinstance(conditions, list) else [conditions]
    for key in ("solver_hints", "extraction_notes"):
        if isinstance(raw.get(key), list):
            out[key] = list(raw[key])

    normalized_nodes: List[Dict[str, Any]] = []
    for index, node in enumerate(out["nodes"]):
        item = dict(node) if isinstance(node, Mapping) else {"value": node}
        symbol = str(item.get("symbol") or item.get("id") or item.get("name") or f"node_{index}")
        item["id"] = str(item.get("id") or symbol)
        item["symbol"] = symbol
        item.setdefault("name", symbol)
        item.setdefault("node_type", "quantity")
        item.setdefault("role", "given" if item.get("value") is not None else "derived")
        item.setdefault("value", item.get("raw_value"))
        item.setdefault("raw_value", item.get("value"))
        item.setdefault("unit", None)
        item.setdefault("dimension", None)
        item.setdefault("source", "extracted node")
        item.setdefault("evidence", item.get("source") or "extracted node")
        item.setdefault("confidence", 1.0)
        normalized_nodes.append(item)
    out["nodes"] = normalized_nodes

    normalized_edges: List[Dict[str, Any]] = []
    for index, edge in enumerate(out["edges"]):
        item = dict(edge) if isinstance(edge, Mapping) else {"rhs": edge}
        item["id"] = str(item.get("id") or f"edge_{index}")
        item["kind"] = str(item.get("kind") or "definition")
        item.setdefault("intent", item["id"])
        item.setdefault("operation", "solve" if item["kind"] == "constraint" else "evaluate")
        item.setdefault("lhs", item.get("left", ""))
        item.setdefault("rhs", item.get("right", ""))
        item.setdefault("operator", item.get("op", "="))
        item.setdefault("inputs", [])
        item.setdefault("outputs", [])
        item.setdefault("tags", [])
        item.setdefault("unit", None)
        item.setdefault("source", "extracted edge")
        item.setdefault("evidence", item.get("source") or "extracted edge")
        item.setdefault("confidence", 1.0)
        item.setdefault("executable", item["kind"] not in {"annotation"})
        normalized_edges.append(item)
    out["edges"] = normalized_edges

    normalized_conditions: List[Dict[str, Any]] = []
    for index, condition in enumerate(out["conditions"]):
        item = dict(condition) if isinstance(condition, Mapping) else {"expr": condition}
        item["id"] = str(item.get("id") or f"condition_{index}")
        item.setdefault("kind", "domain")
        item.setdefault("expr", "")
        item.setdefault("symbols", [])
        item.setdefault("source", "extracted condition")
        item.setdefault("confidence", 1.0)
        normalized_conditions.append(item)
    out["conditions"] = normalized_conditions
    return out


def validate_acg_ir(ir: Mapping[str, Any]) -> List[str]:
    """Validate graph structure while treating domain labels and new operations as extensible."""
    errors: List[str] = []
    if not isinstance(ir, Mapping):
        return ["ACG IR must be an object"]
    target = ir.get("target")
    if not isinstance(target, Mapping):
        errors.append("target must be an object")
        target = {}
    target_symbol = str(target.get("symbol") or target.get("id") or "")
    if not target_symbol or not SYMBOL_RE.fullmatch(target_symbol):
        errors.append("target.symbol must be a valid non-empty Python identifier")
    for key in ("nodes", "edges", "conditions"):
        if not isinstance(ir.get(key), list):
            errors.append(f"{key} must be a list")
    declared = {target_symbol} if target_symbol else set()
    for index, node in enumerate(ir.get("nodes", [])):
        if not isinstance(node, Mapping):
            errors.append(f"node[{index}] must be an object")
            continue
        symbol = str(node.get("symbol") or node.get("id") or "")
        if not symbol or not SYMBOL_RE.fullmatch(symbol):
            errors.append(f"node[{index}].symbol must be a valid identifier")
        else:
            declared.add(symbol)
        if node.get("node_type") not in (None, *ACG_NODE_TYPES):
            errors.append(f"node[{index}].node_type is invalid")
    for index, edge in enumerate(ir.get("edges", [])):
        if not isinstance(edge, Mapping):
            errors.append(f"edge[{index}] must be an object")
            continue
        if not str(edge.get("id") or "").strip():
            errors.append(f"edge[{index}] missing id")
        kind = str(edge.get("kind") or "")
        if kind not in ACG_EDGE_KINDS:
            # Unknown kinds are warnings in the research artifact, not pipeline-fatal errors.
            errors.append(f"edge[{index}] uses extensible kind {kind!r}")
        for field in ("lhs", "rhs", "operator"):
            if field not in edge:
                errors.append(f"edge[{index}] missing field: {field}")
        for field in ("lhs", "rhs"):
            value = str(edge.get(field) or "")
            if "__" in value:
                errors.append(f"edge[{index}].{field} contains unsupported dunder characters")
        lhs = str(edge.get("lhs") or "").strip()
        if SYMBOL_RE.fullmatch(lhs):
            declared.add(lhs)
    for index, condition in enumerate(ir.get("conditions", [])):
        if not isinstance(condition, Mapping):
            errors.append(f"condition[{index}] must be an object")
            continue
        if not str(condition.get("expr") or "").strip():
            errors.append(f"condition[{index}].expr must be non-empty")
    return errors


def legacy_to_acg_ir(ir: Mapping[str, Any]) -> Dict[str, Any]:
    """Lift the existing relation IR into ACG without changing its semantics."""
    source = normalize_ir_shape(ir)
    out = empty_acg_ir()
    target = dict(source.get("target_unknown", {}))
    required = dict(source.get("required_output", {}))
    out["target"] = {
        "id": str(target.get("symbol") or "answer"),
        "name": target.get("name") or target.get("symbol") or "answer",
        "symbol": str(target.get("symbol") or "answer"),
        "unit": target.get("unit"),
        "dimension": target.get("dimension"),
        "output_type": required.get("type", "number"),
        "precision": required.get("precision", "exact"),
        "target_count": required.get("target_count", 1),
    }
    for given in source.get("givens", []):
        if not isinstance(given, Mapping):
            continue
        out["nodes"].append({
            "id": str(given.get("symbol") or given.get("name") or "given"),
            "symbol": str(given.get("symbol") or given.get("name") or "given"),
            "name": given.get("name"), "node_type": "quantity",
            "value": given.get("value"), "raw_value": given.get("value"),
            "unit": given.get("unit"), "role": "given",
            "source": given.get("source", "given"), "evidence": given.get("source", "given"),
            "confidence": 1.0,
        })
    for relation in source.get("relations", []):
        if not isinstance(relation, Mapping):
            continue
        kind = str(relation.get("kind") or "definition")
        graph_kind = "constraint" if relation.get("operator") not in {"=", "=="} else ("annotation" if kind == "general" else "definition")
        out["edges"].append({
            "id": relation.get("id"), "kind": graph_kind, "intent": kind,
            "operation": kind, "lhs": relation.get("lhs", ""), "rhs": relation.get("rhs", ""),
            "operator": relation.get("operator", "="), "inputs": relation.get("symbols", []),
            "outputs": [relation.get("lhs")] if relation.get("lhs") else [],
            "unit": relation.get("unit"), "range": relation.get("range"),
            "source": relation.get("source", "relation"), "evidence": relation.get("evidence", "relation"),
            "confidence": relation.get("confidence", 1.0), "executable": graph_kind != "annotation",
        })
    out["conditions"] = [dict(item) if isinstance(item, Mapping) else {"expr": item} for item in source.get("conditions", [])]
    out["extraction_notes"] = list(source.get("extraction_notes", []))
    return normalize_acg_shape(out)


def acg_to_legacy_ir(acg: Mapping[str, Any]) -> ProblemIR:
    """Project ACG into the current verifier's relation contract."""
    graph = normalize_acg_shape(acg)
    target = graph["target"]
    relations: List[Dict[str, Any]] = []
    for edge in graph["edges"]:
        if edge.get("kind") == "annotation" and not edge.get("executable", False):
            continue
        relations.append({
            "id": edge.get("id"),
            "kind": edge.get("intent") or edge.get("kind") or "equation",
            "lhs": edge.get("lhs", ""), "rhs": edge.get("rhs", ""),
            "operator": edge.get("operator", "="), "unit": edge.get("unit"),
            "range": edge.get("range"), "source": edge.get("source", "edge"),
            "evidence": edge.get("evidence", edge.get("source", "edge")),
            "confidence": edge.get("confidence", 1.0),
        })
    return normalize_ir_shape({
        "target_unknown": {"name": target.get("name"), "symbol": target.get("symbol"), "unit": target.get("unit"), "dimension": target.get("dimension")},
        "givens": [
            {"name": node.get("name"), "symbol": node.get("symbol"), "value": node.get("value"), "unit": node.get("unit"), "role": node.get("role", "constant"), "source": node.get("source", "node")}
            for node in graph["nodes"]
        ],
        "relations": relations,
        "conditions": graph["conditions"],
        "required_output": {"type": target.get("output_type", "number"), "unit": target.get("unit"), "precision": target.get("precision", "exact"), "digits": None, "target_count": target.get("target_count", 1)},
        "extraction_notes": graph.get("extraction_notes", []),
    })


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
