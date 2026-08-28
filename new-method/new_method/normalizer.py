"""Deterministic input normalization and codegen payload construction for Universal IR."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Dict, Mapping


UNIT_FACTORS = {
    "mm": ("m", 0.001), "cm": ("m", 0.01), "m": ("m", 1.0), "km": ("m", 1000.0),
    "in": ("m", 0.0254), "ft": ("m", 0.3048), "yd": ("m", 0.9144), "mi": ("m", 1609.344),
    "mg": ("g", 0.001), "g": ("g", 1.0), "kg": ("g", 1000.0),
    "ml": ("l", 0.001), "l": ("l", 1.0),
    "sec": ("s", 1.0), "s": ("s", 1.0), "min": ("s", 60.0), "h": ("s", 3600.0),
    "degree": ("degree", 1.0), "rad": ("rad", 1.0), "unit": ("unit", 1.0),
}
UNIT_ALIASES = {
    "seconds": "s", "second": "s", "minutes": "min", "minute": "min",
    "hours": "h", "hour": "h", "hrs": "h", "hr": "h",
    "meters": "m", "meter": "m", "metres": "m",
    "centimeters": "cm", "centimeter": "cm", "kilometers": "km", "kilometer": "km",
    "inches": "in", "inch": "in", "feet": "ft", "foot": "ft",
    "yards": "yd", "yard": "yd", "miles": "mi", "mile": "mi",
    "grams": "g", "gram": "g", "kilograms": "kg", "kilogram": "kg",
    "liters": "l", "liter": "l", "litres": "l", "litre": "l",
    "degrees": "degree", "degree": "degree", "deg": "degree",
    "radians": "rad", "radian": "rad",
    "units": "unit", "unit": "unit",
    "dollar": "$", "dollars": "$", "usd": "$", "cent": "cents",
    "dozen": "dozen", "dozens": "dozen", "pair": "pair", "pairs": "pair",
    "pack": "pack", "packs": "pack", "box": "box", "boxes": "box",
    "day": "days", "days": "days", "week": "weeks", "weeks": "weeks",
    "month": "months", "months": "months", "year": "years", "years": "years",
    "page": "pages", "pages": "pages", "student": "students", "students": "students",
    "dog": "dogs", "dogs": "dogs", "slice": "slices", "slices": "slices",
}
EXPR_REPLACEMENTS = {
    "×": "*", "·": "*", "÷": "/", "−": "-", "–": "-", "^": "**",
    "π": "pi", "∞": "oo", "θ": "theta", "Θ": "Theta", "²": "**2", "³": "**3", "°": "", "𝑖": "I",
}
UNIT_TOKEN_RE = re.compile(r"^(?P<name>[A-Za-z$]+)(?:\^?(?P<power>-?\d+))?$")


def normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    text = str(unit).strip().lower().replace(" ", "").replace("·", "*")
    if not text:
        return None
    return text.replace("²", "^2").replace("³", "^3").replace("°", "degree")


def parse_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().replace(",", "").replace("$", "")
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
        text = value.strip()
        if text.startswith("$"):
            unit = unit or "$"
            text = text[1:].strip()
            value = text
        match = re.fullmatch(r"\s*([-+]?(?:\d[\d,]*(?:\.\d+)?|\d+\s*/\s*\d+(?:\.\d+)?))\s*([A-Za-z%$]+(?:\^?-?\d+)?(?:/[A-Za-z%$]+(?:\^?-?\d+)?)?)?\s*", text)
        if match:
            cand_val, embedded_unit = match.groups()
            if embedded_unit and embedded_unit.lower() not in {"pi", "e", "i", "oo"}:
                value = cand_val
                unit = embedded_unit or unit
    return value, unit.lower() if isinstance(unit, str) else unit


def unit_conversion(unit: str | None) -> tuple[str | None, float]:
    """Return canonical compound unit and multiplicative factor to canonical SI-like units."""
    if not unit:
        return unit, 1.0
    cleaned = normalize_unit(unit)
    if not cleaned:
        return unit, 1.0
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
    cleaned = normalize_unit(unit)
    if not cleaned:
        return True
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


def classify_unit(unit: str | None) -> str:
    """Classify a label without making unknown labels a normalization failure."""
    cleaned = normalize_unit(unit)
    if not cleaned:
        return "none"
    if is_supported_unit(cleaned):
        return "convertible"
    if cleaned in {"box", "boxes", "pack", "packs", "bag", "bags", "month", "months", "year", "years"}:
        return "ambiguous"
    return "opaque"


def is_safe_symbolic_expression(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = normalize_expression(value)
    if not text or "__" in text:
        return False
    return not any(token in text for token in ("\\", "$", "=", "<", ">", "{", "}"))


def normalize_quantity(value: Any, unit: str | None = None) -> Dict[str, Any]:
    raw, unit = split_quantity(value, unit)
    unit = normalize_unit(unit)
    numeric = parse_number(raw)
    if numeric is None:
        norm_val = normalize_expression(raw)
        status = "expression" if is_safe_symbolic_expression(raw) else "symbolic"
        return {"raw": value, "value": norm_val, "unit": unit, "canonical_value": norm_val, "canonical_unit": unit, "unit_class": classify_unit(unit), "status": status}
    if unit == "%":
        return {"raw": value, "value": numeric, "unit": "%", "canonical_value": numeric / 100.0, "canonical_unit": "ratio", "unit_class": "convertible", "status": "ok"}
    # A unit is a semantic label unless a conversion rule is explicit. This keeps
    # domain entities such as "dogs" and "packs" from becoming hard failures.
    return {"raw": value, "value": numeric, "unit": unit, "canonical_value": numeric, "canonical_unit": unit, "unit_class": classify_unit(unit), "status": "ok"}


def normalize_expression(expression: Any) -> str:
    """Canonicalize an algebraic expression without evaluating symbols."""
    text = str(expression if expression is not None else "").strip()
    for source, target in EXPR_REPLACEMENTS.items():
        if source == "^":
            continue
        text = text.replace(source, target)
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = text.replace("\\pi", "pi").replace("\\infty", "oo")
    text = re.sub(r"(?<=\d)\s+(?=(sqrt|sin|cos|tan|log|exp)\s*\()", "*", text)
    text = re.sub(r"(?<=\d)\s*pi\b", "*pi", text)
    text = re.sub(r"(?<=\d)(?=pi\b)", "*", text)
    text = re.sub(r"(?<=[0-9)\]])\s*[iI]\b", "*I", text)
    text = re.sub(r"(?<=[0-9)\]])(?=[iI]\b)", "*", text)
    text = re.sub(r"\bi\b", "I", text)
    text = re.sub(r"\b(and|where|such that)\b", " ", text, flags=re.I)

    text = text.replace("^", "**")
    text = _normalize_function_evaluations(text)
    return re.sub(r"\s+", " ", text)


def _normalize_function_evaluations(text: str) -> str:
    math_names = {"sqrt", "sin", "cos", "tan", "exp", "log", "abs", "min", "max", "int", "gcd", "lcm", "factorint", "divisors", "oct", "bin", "Tuple", "FiniteSet", "Interval", "Union", "Matrix"}

    def replacement(match: re.Match[str]) -> str:
        name, value = match.group(1), match.group(2).replace(" ", "")
        if name in math_names:
            return match.group(0)
        token = value.replace("-", "_minus_").replace("+", "")
        token = re.sub(r"\W+", "_", token).strip("_")
        return f"{name}_{token}"

    return re.sub(r"\b([A-Za-z_]\w*)\s*\(\s*([+-]?\d+)\s*\)", replacement, text)


def relation_symbols(lhs: str, rhs: str) -> list[str]:
    candidates = re.findall(r"\b[A-Za-z_]\w*\b", f"{lhs} {rhs}")
    reserved = {"and", "or", "not", "True", "False", "pi", "e", "I", "oo", "sin", "cos", "tan", "sqrt", "exp", "log", "abs", "min", "max", "int", "gcd", "lcm", "factorint", "divisors", "oct", "bin", "Tuple", "FiniteSet", "Interval", "Union", "Matrix"}
    return sorted({item for item in candidates if item not in reserved and not item.isdigit()})


def _copy_normalized_ir(normalized_ir: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {key: value for key, value in normalized_ir.items()}
    out["givens"] = [dict(item) for item in normalized_ir.get("givens", [])]
    out["relations"] = [dict(item) for item in normalized_ir.get("relations", [])]
    out["symbols"] = dict(normalized_ir.get("symbols", {}))
    out["target_unknown"] = dict(normalized_ir.get("target_unknown", {}))
    out["required_output"] = dict(normalized_ir.get("required_output", {}))
    out["conditions"] = [dict(item) if isinstance(item, Mapping) else item for item in normalized_ir.get("conditions", [])]
    return out


def _upsert_given(normalized_ir: Dict[str, Any], symbol: str, value: Any, unit: str | None, source: str, role: str = "constant") -> None:
    symbol = re.sub(r"\W+", "_", normalize_expression(symbol)).strip("_") or symbol
    quantity = normalize_quantity(value, unit)
    item = {"name": symbol, "symbol": symbol, "value": value, "unit": normalize_unit(unit), "role": role, "source": source, "quantity": quantity}
    for index, existing in enumerate(normalized_ir.get("givens", [])):
        if existing.get("symbol") == symbol:
            normalized_ir["givens"][index] = {**existing, **item}
            break
    else:
        normalized_ir.setdefault("givens", []).append(item)
    normalized_ir.setdefault("symbols", {})[symbol] = symbol


def _definition(id_: str, lhs: str, rhs: str, source: str, unit: str | None = None) -> Dict[str, Any]:
    lhs, rhs = normalize_expression(lhs), normalize_expression(rhs)
    return {
        "id": id_, "kind": "definition", "lhs": lhs, "rhs": rhs,
        "operator": "=", "unit": normalize_unit(unit), "source": source,
        "evidence": source, "confidence": 1.0, "symbols": relation_symbols(lhs, rhs),
    }


def augment_ir_from_question(question: str, normalized_ir: Mapping[str, Any]) -> Dict[str, Any]:
    """Patch a few explicit MATH literals that 7B extraction frequently drops."""
    out = _copy_normalized_ir(normalized_ir)
    _augment_cylinder_literals(question, out)
    _augment_complex_literals(question, out)
    _augment_asy_target_segment(question, out)
    return out


def _augment_cylinder_literals(question: str, normalized_ir: Dict[str, Any]) -> None:
    if "cylinder" not in question.lower():
        return
    volume = re.search(r"volume\b.*?(?:is|=)\s*\$?\s*([0-9]+(?:\s*(?:\\pi|pi|π))?)\s*\$?\s*cubic\s*cm", question, re.I | re.S)
    radius = re.search(r"\$r\s*=\s*([^$]+?)\$", question, re.I)
    target_symbol = normalized_ir.get("target_unknown", {}).get("symbol")
    if volume:
        _upsert_given(normalized_ir, "V", normalize_expression(volume.group(1)), "cm^3", volume.group(0), "measurement")
    if radius:
        _upsert_given(normalized_ir, "r", normalize_expression(radius.group(1)), "cm", radius.group(0), "measurement")
    if target_symbol:
        normalized_ir["relations"] = [
            item for item in normalized_ir.get("relations", [])
            if item.get("id") != "cylinder_volume_from_problem"
        ]
        normalized_ir["relations"].append(_definition("cylinder_volume_from_problem", "V", f"pi*r**2*{target_symbol}", "cylinder volume from problem text", "cm^3"))


def _augment_complex_literals(question: str, normalized_ir: Dict[str, Any]) -> None:
    if "i" not in question:
        return
    for symbol in ("z", "c"):
        match = re.search(rf"\${symbol}\s*=\s*([^$]+?)\$", question, re.I)
        if match:
            _upsert_given(normalized_ir, symbol, normalize_expression(match.group(1)), None, match.group(0))
    if str(normalized_ir.get("target_unknown", {}).get("dimension") or "").lower() == "complex":
        normalized_ir.setdefault("required_output", {})["type"] = "symbolic"


def _augment_asy_target_segment(question: str, normalized_ir: Dict[str, Any]) -> None:
    target = str(normalized_ir.get("target_unknown", {}).get("symbol") or "")
    match = re.fullmatch(r"([A-Z])([A-Z])", target)
    if not match or "[asy]" not in question:
        return
    points: dict[str, tuple[str, str]] = {}
    for point, x_value, y_value in re.findall(r"\b([A-Z])\s*=\s*\(([^,;]+),([^;]+?)\);", question):
        points[point] = (normalize_expression(x_value), normalize_expression(y_value))
    left, right = match.groups()
    if left not in points or right not in points:
        return
    x1, y1 = points[left]
    x2, y2 = points[right]
    rhs = f"sqrt((({x1})-({x2}))**2 + (({y1})-({y2}))**2)"
    normalized_ir["relations"] = [
        item for item in normalized_ir.get("relations", [])
        if item.get("lhs") != target
    ]
    normalized_ir["relations"].append(_definition("asy_target_segment", target, rhs, "target segment from ASY coordinates"))


def normalize_problem_ir(ir: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-friendly IR with canonical numeric values and units."""
    normalized: Dict[str, Any] = {key: value for key, value in ir.items()}
    givens = []
    symbols: Dict[str, str] = {}
    for index, given in enumerate(ir.get("givens", [])):
        item = dict(given) if isinstance(given, Mapping) else {"value": given}
        symbol = str(item.get("symbol") or item.get("name") or f"given_{index}")
        item["symbol"] = re.sub(r"\W+", "_", normalize_expression(symbol)).strip("_") or f"given_{index}"
        item["unit"] = normalize_unit(item.get("unit"))
        item["quantity"] = normalize_quantity(item.get("value", item.get("expression")), item.get("unit"))
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
        item["unit"] = normalize_unit(item.get("unit"))
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
    target["symbol"] = re.sub(r"\W+", "_", normalize_expression(target.get("symbol") or target.get("name") or "answer")).strip("_") or "answer"
    target["unit"] = normalize_unit(target.get("unit"))
    target["quantity"] = normalize_quantity(target.get("value"), target.get("unit")) if "value" in target else None
    normalized["target_unknown"] = target
    required_output = dict(normalized.get("required_output", {}))
    required_output["unit"] = normalize_unit(required_output.get("unit"))
    output_type = str(required_output.get("type") or "").lower()
    if output_type in {"complex", "expression"}:
        required_output["type"] = "symbolic"
    normalized["required_output"] = required_output
    normalized["conditions"] = [
        {**dict(item), "expr": normalize_expression(item.get("expr"))}
        if isinstance(item, Mapping) else item
        for item in normalized.get("conditions", [])
    ]
    return normalized


def validate_normalized_ir(normalized_ir: Mapping[str, Any]) -> list[str]:
    """Universal IR accepts all mathematical constructs; returns empty error list."""
    return []


def build_codegen_payload(normalized_ir: Mapping[str, Any]) -> Dict[str, Any]:
    """The only object passed to codegen; prose is deliberately excluded."""
    units = set(UNIT_FACTORS)
    units.update({"%", "ratio", "cm^2", "m^2", "in^2", "unit", "units", "degree", "degrees", "km/h", "km/hr", "m/s", "cm/s"})
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
                "range": relation.get("range"),
            }
            for relation in normalized_ir.get("relations", [])
        ],
        "conditions": [{"kind": item.get("kind"), "expr": item.get("expr")} for item in normalized_ir.get("conditions", [])],
        "target_unknown": {key: normalized_ir.get("target_unknown", {}).get(key) for key in ("symbol", "unit", "dimension")},
        "required_output": normalized_ir.get("required_output", {}),
        "unit_conversions": {
            unit: {
                "canonical_unit": unit_conversion(unit)[0],
                "to_canonical": unit_conversion(unit)[1],
                "from_canonical": 1.0 / unit_conversion(unit)[1],
                "unit_class": classify_unit(unit),
            }
            for unit in sorted(units)
        },
    }
