"""Deterministic input normalization and codegen payload construction."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Dict, Mapping


UNIT_FACTORS = {
    "mm": ("m", 0.001), "cm": ("m", 0.01), "m": ("m", 1.0), "km": ("m", 1000.0),
    "mg": ("g", 0.001), "g": ("g", 1.0), "kg": ("g", 1000.0),
    "ml": ("l", 0.001), "l": ("l", 1.0),
    "sec": ("s", 1.0), "s": ("s", 1.0), "min": ("s", 60.0), "h": ("s", 3600.0),
}
UNIT_ALIASES = {
    "seconds": "s", "second": "s", "minutes": "min", "minute": "min",
    "hours": "h", "hour": "h", "meters": "m", "meter": "m", "metres": "m",
    "centimeters": "cm", "centimeter": "cm", "kilometers": "km", "kilometer": "km",
    "grams": "g", "gram": "g", "kilograms": "kg", "kilogram": "kg",
    "liters": "l", "liter": "l", "litres": "l", "litre": "l",
}
EXPR_REPLACEMENTS = {"×": "*", "·": "*", "÷": "/", "−": "-", "–": "-", "^": "**", "π": "pi"}
UNIT_TOKEN_RE = re.compile(r"^(?P<name>[A-Za-z]+)(?:\^?(?P<power>-?\d+))?$")


def parse_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().replace(",", "")
    if text.endswith("%"):
        number = parse_number(text[:-1])
        return None if number is None else number / 100.0
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(Fraction(match.group(1)) / Fraction(match.group(2)))
    try:
        return float(text)
    except ValueError:
        return None


def split_quantity(value: Any, unit: str | None = None) -> tuple[Any, str | None]:
    if isinstance(value, Mapping):
        unit = value.get("unit", unit)
        value = value.get("value", value.get("amount"))
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([-+]?(?:\d[\d,]*(?:\.\d+)?|\d+\s*/\s*\d+(?:\.\d+)?))\s*([A-Za-z%]+(?:\^?-?\d+)?(?:/[A-Za-z%]+(?:\^?-?\d+)?)?)?\s*", value)
        if match:
            value, embedded_unit = match.groups()
            unit = embedded_unit or unit
    return value, unit.lower() if isinstance(unit, str) else unit


def unit_conversion(unit: str | None) -> tuple[str | None, float]:
    """Return canonical compound unit and multiplicative factor to canonical SI-like units."""
    if not unit:
        return unit, 1.0
    cleaned = str(unit).strip().lower().replace(" ", "").replace("·", "*")
    if cleaned == "%":
        return "ratio", 0.01
    if cleaned == "ratio":
        return "ratio", 1.0
    numerator, *denominators = cleaned.split("/")
    parts = []
    factor = 1.0
    for side, sign in [(numerator, 1)] + [(item, -1) for item in denominators]:
        for token in side.split("*"):
            match = UNIT_TOKEN_RE.fullmatch(token)
            if not match:
                return unit, 1.0
            name = UNIT_ALIASES.get(match.group("name"), match.group("name"))
            if name not in UNIT_FACTORS:
                return unit, 1.0
            power = int(match.group("power") or 1)
            canonical, base_factor = UNIT_FACTORS[name]
            factor *= (base_factor ** (power * sign))
            parts.append((canonical, power * sign))
    compact = []
    for canonical, power in parts:
        if power == 1:
            compact.append(canonical)
        elif power != 0:
            compact.append(f"{canonical}^{power}")
    canonical_unit = "*".join(item for item in compact if not item.startswith("/"))
    negative = [item for canonical, power in parts for item in ([canonical] if power < 0 else [])]
    if negative:
        positive = [canonical if power == 1 else f"{canonical}^{power}" for canonical, power in parts if power > 0]
        denominator = "*".join(canonical if abs(power) == 1 else f"{canonical}^{abs(power)}" for canonical, power in parts if power < 0)
        canonical_unit = f"{'*'.join(positive) or '1'}/{denominator}"
    return canonical_unit or unit, factor


def is_supported_unit(unit: str | None) -> bool:
    if unit is None or str(unit).strip() == "":
        return True
    cleaned = str(unit).strip().lower().replace(" ", "").replace("·", "*")
    if cleaned in {"%", "ratio"}:
        return True
    for side in cleaned.split("/"):
        if not side:
            return False
        for token in side.split("*"):
            match = UNIT_TOKEN_RE.fullmatch(token)
            if not match:
                return False
            name = UNIT_ALIASES.get(match.group("name"), match.group("name"))
            if name not in UNIT_FACTORS:
                return False
    return True


def normalize_quantity(value: Any, unit: str | None = None) -> Dict[str, Any]:
    raw, unit = split_quantity(value, unit)
    numeric = parse_number(raw)
    if numeric is None:
        return {"raw": value, "value": None, "unit": unit, "canonical_value": None, "canonical_unit": unit, "status": "non_numeric"}
    if not is_supported_unit(unit):
        return {"raw": value, "value": numeric, "unit": unit, "canonical_value": None, "canonical_unit": unit, "status": "unknown_unit"}
    if unit == "%":
        return {"raw": value, "value": numeric, "unit": "%", "canonical_value": numeric / 100.0, "canonical_unit": "ratio", "status": "ok"}
    canonical_unit, factor = unit_conversion(unit)
    return {"raw": value, "value": numeric, "unit": unit, "canonical_value": numeric * factor, "canonical_unit": canonical_unit, "status": "ok"}


def normalize_expression(expression: Any) -> str:
    """Canonicalize an algebraic expression without evaluating symbols."""
    text = str(expression if expression is not None else "").strip()
    # Convert unit-bearing literals before translating ^; otherwise ``4 cm^2``
    # would be misread as ``(4 cm) ** 2``.
    for source, target in EXPR_REPLACEMENTS.items():
        if source == "^":
            continue
        text = text.replace(source, target)
    text = re.sub(r"\b(and|where|such that)\b", " ", text, flags=re.I)

    def replace_quantity(match: re.Match[str]) -> str:
        quantity = normalize_quantity(match.group(0))
        value = quantity.get("canonical_value")
        return str(value) if value is not None else match.group(0)

    text = re.sub(r"[-+]?(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+(?:\.\d+)?)\s*[A-Za-z%]+(?:\^?-?\d+)?(?:/[A-Za-z%]+(?:\^?-?\d+)?)?(?![A-Za-z0-9_])", replace_quantity, text, flags=re.I)
    text = text.replace("^", "**")
    return re.sub(r"\s+", " ", text)


def relation_symbols(lhs: str, rhs: str) -> list[str]:
    candidates = re.findall(r"\b[A-Za-z_]\w*\b", f"{lhs} {rhs}")
    reserved = {"and", "or", "not", "True", "False", "pi", "e", "sin", "cos", "tan", "sqrt", "exp", "log"}
    return sorted({item for item in candidates if item not in reserved and not item.isdigit()})


def normalize_problem_ir(ir: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-friendly IR with canonical numeric values and units."""
    normalized: Dict[str, Any] = {key: value for key, value in ir.items()}
    givens = []
    symbols: Dict[str, str] = {}
    for index, given in enumerate(ir.get("givens", [])):
        item = dict(given) if isinstance(given, Mapping) else {"value": given}
        symbol = str(item.get("symbol") or item.get("name") or f"given_{index}")
        item["symbol"] = re.sub(r"\W+", "_", symbol).strip("_") or f"given_{index}"
        item["quantity"] = normalize_quantity(item.get("value", item.get("expression")), item.get("unit"))
        if item["quantity"].get("status") == "non_numeric" and item.get("role") in {"parameter", "variable"}:
            symbolic_value = str(item.get("value") or item["symbol"]).strip()
            item["quantity"] = {"raw": item.get("value"), "value": symbolic_value, "unit": item.get("unit"), "canonical_value": symbolic_value, "canonical_unit": item.get("unit"), "status": "symbolic"}
        symbols[item["symbol"]] = item["symbol"]
        givens.append(item)
    normalized["givens"] = givens
    normalized["symbols"] = symbols
    relations = []
    for index, relation in enumerate(ir.get("relations", [])):
        if not isinstance(relation, Mapping):
            continue
        item = dict(relation)
        item["id"] = str(item.get("id") or f"relation_{index}")
        item["kind"] = str(item.get("kind") or ("inequality" if item.get("operator") in {"<", "<=", ">", ">="} else "equation"))
        item["operator"] = str(item.get("operator") or "=")
        item["lhs"] = normalize_expression(item.get("lhs"))
        item["rhs"] = normalize_expression(item.get("rhs"))
        item["symbols"] = relation_symbols(item["lhs"], item["rhs"])
        try:
            item["confidence"] = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            item["confidence"] = 1.0
        relations.append(item)
        if item["kind"] == "definition" and re.fullmatch(r"[A-Za-z_]\w*", item["lhs"]):
            symbols[item["lhs"]] = item["lhs"]
    normalized["relations"] = relations
    target = dict(ir.get("target_unknown", {}))
    target["symbol"] = re.sub(r"\W+", "_", str(target.get("symbol") or target.get("name") or "answer")).strip("_") or "answer"
    target["quantity"] = normalize_quantity(target.get("value"), target.get("unit")) if "value" in target else None
    normalized["target_unknown"] = target
    return normalized


def validate_normalized_ir(normalized_ir: Mapping[str, Any]) -> list[str]:
    """Reject quantities/units that could not be deterministically normalized."""
    errors: list[str] = []
    for index, given in enumerate(normalized_ir.get("givens", [])):
        quantity = given.get("quantity", {})
        if quantity.get("status") == "unknown_unit":
            errors.append(f"given[{index}] has unsupported unit: {quantity.get('unit')}")
        elif quantity.get("status") == "symbolic" and given.get("role") in {"parameter", "variable"}:
            pass
        elif quantity.get("canonical_value") is None:
            errors.append(f"given[{index}] is not a concrete numeric quantity")
    for index, relation in enumerate(normalized_ir.get("relations", [])):
        if relation.get("unit") and not is_supported_unit(relation.get("unit")):
            errors.append(f"relation[{index}] has unsupported unit: {relation.get('unit')}")
    for path, unit in (
        ("target_unknown.unit", normalized_ir.get("target_unknown", {}).get("unit")),
        ("required_output.unit", normalized_ir.get("required_output", {}).get("unit")),
    ):
        if unit and not is_supported_unit(unit):
            errors.append(f"{path} is unsupported: {unit}")
    return errors


def build_codegen_payload(normalized_ir: Mapping[str, Any]) -> Dict[str, Any]:
    """The only object passed to codegen; prose is deliberately excluded."""
    units = set(UNIT_FACTORS)
    units.update({"%", "ratio", "cm^2", "m^2", "km/h", "m/s", "cm/s"})
    for given in normalized_ir.get("givens", []):
        units.update(filter(None, [given.get("unit"), given.get("quantity", {}).get("canonical_unit")]))
    units.update(filter(None, [normalized_ir.get("target_unknown", {}).get("unit"), normalized_ir.get("required_output", {}).get("unit")]))
    return {
        "givens": [
            {
                "symbol": given["symbol"],
                "value": given["quantity"]["canonical_value"],
                "unit": given["quantity"]["canonical_unit"],
            }
            for given in normalized_ir.get("givens", [])
        ],
        "relations": [
            {
                "id": relation.get("id"),
                "kind": relation.get("kind", "equation"),
                "lhs": relation.get("lhs"),
                "rhs": relation.get("rhs"),
                "operator": relation.get("operator", "="),
                "symbols": relation.get("symbols", []),
                "unit": relation.get("unit"),
            }
            for relation in normalized_ir.get("relations", [])
        ],
        "conditions": [{"kind": item.get("kind"), "expr": item.get("expr")} for item in normalized_ir.get("conditions", [])],
        "target_unknown": {key: normalized_ir.get("target_unknown", {}).get(key) for key in ("symbol", "unit", "dimension")},
        "required_output": normalized_ir.get("required_output", {}),
        "unit_conversions": {
            unit: {"canonical_unit": unit_conversion(unit)[0], "to_canonical": unit_conversion(unit)[1], "from_canonical": 1.0 / unit_conversion(unit)[1]}
            for unit in sorted(units) if is_supported_unit(unit)
        },
    }
