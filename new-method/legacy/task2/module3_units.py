#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module 3: chuẩn hóa giá trị và đơn vị về SI cho Task 2.

File này cố ý chỉ làm normalization, không suy luận công thức hay thêm giả định
vật lý. Nhờ vậy lỗi ở Module 3 dễ khoanh vùng: nếu sai thì thường là parse số
hoặc bảng đổi đơn vị, không phải do planner/codegen.

Không làm các việc sau:
- suy luận công thức
- suy luận hình học
- thêm physics hint
- đoán target unit
- thêm hằng số
- đổi nghĩa conditions

API chính:
    process_module3(module2_json)

Input mong đợi từ Module 2:
{
  "given": [
    {"name": "...", "symbol": "...", "value": ..., "unit": "..."}
  ],
  "conditions": ["..."],
  "target": [
    {"name": "...", "symbol": "..."}
  ]
}

Output:
{
  "given": [
    {
      "name": "...",
      "symbol": "...",
      "value": SI_value_or_original_symbolic,
      "unit": SI_unit,
      "original_value": ...,
      "original_unit": "...",
      "is_numeric": true/false,
      "status": "ok" | "unknown_unit" | "symbolic_value"
    }
  ],
  "conditions": [...],
  "target": [...]
}
"""


import ast
import copy
import math
import operator as op
import re
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Safe numeric parser
# ============================================================

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

_SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁻": "-",
    "⁺": "+",
})


def _safe_eval_number_expr(expr: str) -> float:
    """Safely evaluate simple arithmetic numeric expressions."""
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Num):  # compatibility for older Python
            return float(node.n)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Invalid constant")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed")
            return _ALLOWED_OPERATORS[op_type](_eval(node.left), _eval(node.right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPERATORS:
                raise ValueError("Unary operator not allowed")
            return _ALLOWED_OPERATORS[op_type](_eval(node.operand))

        raise ValueError("Unsupported expression")

    tree = ast.parse(expr, mode="eval")
    return float(_eval(tree))


def parse_numeric_value(value: Any) -> Tuple[Optional[float], bool]:
    """
    Parse numeric value safely.

    Supported:
    - 3
    - "3"
    - "2,5"
    - "6e-8"
    - "6 × 10^-8"
    - "24.45 × 10^-3"
    - "1/2"

    Symbolic values such as "U", "2U", "250*sin(1000*t)" are kept unchanged.
    """
    if value is None or isinstance(value, bool):
        return None, False

    if isinstance(value, (int, float)):
        number = float(value)
        return (number, True) if math.isfinite(number) else (None, False)

    s = str(value).strip()
    if not s:
        return None, False

    # Dataset hay viết scientific notation bằng chữ x: "16 x 10^-8".
    s = re.sub(r"(?<=\d)\s*[xX]\s*10", "*10", s)

    # Keep symbolic expressions symbolic.
    # Remove scientific notation first so "6e-8" is not treated as symbolic.
    tmp = re.sub(r"[eE][-+]?\d+", "", s)
    if re.search(r"[A-Za-zα-ωΑ-Ω]", tmp):
        return None, False

    s = s.translate(_SUPERSCRIPT_MAP)
    s = s.replace(",", ".")
    s = s.replace("×", "*").replace("·", "*").replace("⋅", "*")
    s = s.replace("^", "**")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", "", s)
    s = re.sub(
        r"([-+]?\d+(?:\.\d+)?)\*10(?:\*\*)?([-+]?\d+)",
        r"\1e\2",
        s,
        flags=re.IGNORECASE,
    )

    if not re.fullmatch(r"[-+0-9.eE*/()]+", s):
        return None, False

    try:
        return _safe_eval_number_expr(s), True
    except Exception:
        try:
            return float(s), True
        except Exception:
            return None, False


# ============================================================
# Unit normalization
# ============================================================

DIMENSIONLESS_UNITS = {
    "",
    "-",
    "—",
    "none",
    "None",
    "null",
    "dimensionless",
    "unitless",
    "no_unit",
    "lần",
    "lan",
}


def canonicalize_unit_text(unit: Any, collapse_electric_field: bool = False) -> str:
    """Chuẩn hóa text đơn vị về notation BTC/pipeline dùng chung.

    Hàm này là cửa vào chung cho unit extraction, SI normalizer, verifier và
    submission export. `collapse_electric_field=True` dùng khi so sánh đơn vị
    để coi `N/C` và `V/m` là tương đương.
    """
    if unit is None:
        return ""

    raw = str(unit).strip()
    if raw in DIMENSIONLESS_UNITS:
        return ""

    u = raw.replace("\\\\", "\\")
    # Thay thế các cụm LaTeX phổ biến trong đơn vị
    u = u.replace("\\cdot", "*").replace("\\times", "*").replace("×", "*").replace("·", "*")
    u = u.replace("\\,", "").replace("\\;", "").replace("\\:", "").replace("\\!", "").replace("\\ ", "")

    # Xử lý \mathrm, \text lồng nhau hoặc nhiều khối
    for _ in range(5):
        new_u = re.sub(r"\\(?:mathrm|text|mathbf)\s*\{\s*([^{}]+?)\s*\}", r"\1", u)
        if new_u == u:
            break
        u = new_u

    u = re.sub(r"\\frac\s*\{\s*([^}]+?)\s*\}\s*\{\s*([^}]+?)\s*\}", r"\1/\2", u)
    u = u.replace("\\over", "/")
    u = u.replace("\\Omega", "ohm")
    u = u.replace("\\%", "%")
    u = u.replace("\\mu", "u")
    u = u.replace("Ω", "ohm").replace("Ω", "ohm")
    u = u.replace("μ", "u").replace("µ", "u")
    u = u.replace("²", "^2").replace("³", "^3")
    u = u.replace("−", "-").replace("–", "-").replace("—", "-")
    u = u.replace("⋅", "*").replace("·", "*")
    u = re.sub(r"\bper\b", "/", u, flags=re.IGNORECASE)
    u = u.replace("{", "").replace("}", "")
    u = re.sub(r"\s+", "", u)
    u = u.replace(".", "*") # Thường dùng dấu chấm thay cho nhân đơn vị: kW.h -> kW*h

    # Case-insensitive mapping for aliases
    u_lower = u.lower()
    aliases = {
        "": "",
        "-": "",
        "none": "",
        "null": "",
        "dimensionless": "",
        "unitless": "",
        "no_unit": "",
        "lan": "",
        "lần": "",

        "%": "%",
        "percent": "%",
        "percentage": "%",

        "degree": "deg",
        "degrees": "deg",
        "độ": "deg",
        "°": "deg",
        "rad": "rad",
        "radian": "rad",
        "radians": "rad",
        "rad/s": "rad/s",
        "rad/s^2": "rad/s^2",

        "°c": "degC",
        "degc": "degC",
        "celsius": "degC",

        "turn": "turn",
        "turns": "turn",
        "vòng": "turn",
        "vong": "turn",

        "sec": "s",
        "second": "s",
        "seconds": "s",
        "min": "min",
        "minute": "min",
        "minutes": "min",
        "hr": "h",
        "hour": "h",
        "hours": "h",

        "m": "m",
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",

        "g": "g",
        "gram": "g",
        "grams": "g",
        "tấn": "ton",
        "tan": "ton",
        "tonne": "ton",

        "ohm": "ohm",
        "ohms": "ohm",
        "kohm": "kohm",
        "mohm": "Mohm",

        "uf": "uF",
        "uc": "uC",
        "uj": "uJ",
        "ua": "uA",
        "uh": "uH",
        "ut": "uT",
        "uwb": "uWb",

        "farad": "F",
        "henry": "H",
        "tesla": "T",
        "volt": "V",
        "ampere": "A",
        "newton": "N",
        "joule": "J",
        "watt": "W",
        "pa": "Pa",
        "kpa": "kPa",
        "mpa": "MPa",

        "n/c": "N/C",
        "v/m": "V/m",
        "nperc": "N/C",
        "vperm": "V/m",

        "kw*h": "kWh",
        "kwh": "kWh",
        "kw.h": "kWh",
        "j*s": "J*s",
        "n*m": "N*m",
        "kg*m/s": "kg*m/s",
        "kg*m/s^2": "N",
    }

    if u_lower in aliases:
        u = aliases[u_lower]
    else:
        # Xử lý các tiền tố viết hoa thường lẫn lộn cho đơn vị chuẩn
        for known in ["F", "C", "J", "A", "H", "T", "V", "N", "W", "Pa", "Wb"]:
            if u_lower == known.lower():
                u = known
                break
            for prefix in ["m", "u", "n", "p", "k", "M", "G"]:
                if u_lower == (prefix + known).lower():
                    u = prefix + known
                    break

    if collapse_electric_field and u in {"N/C", "V/m"}:
        return "E_FIELD"
    return u

def clean_unit(unit: Any) -> str:
    """Normalize unit spelling to table keys."""
    return canonicalize_unit_text(unit)


# unit -> (factor_to_si, si_unit)
UNIT_TABLE: Dict[str, Tuple[float, str]] = {
    # dimensionless / counts / angle
    "": (1.0, ""),
    "%": (0.01, ""),
    "deg": (math.pi / 180.0, "rad"),
    "rad": (1.0, "rad"),
    "degC": (1.0, "degC"),
    "turn": (1.0, ""),

    # length
    "m": (1.0, "m"),
    "km": (1e3, "m"),
    "dm": (1e-1, "m"),
    "cm": (1e-2, "m"),
    "mm": (1e-3, "m"),
    "um": (1e-6, "m"),
    "nm": (1e-9, "m"),

    # area
    "m^2": (1.0, "m^2"),
    "m2": (1.0, "m^2"),
    "km^2": (1e6, "m^2"),
    "km2": (1e6, "m^2"),
    "cm^2": (1e-4, "m^2"),
    "cm2": (1e-4, "m^2"),
    "mm^2": (1e-6, "m^2"),
    "mm2": (1e-6, "m^2"),

    # volume
    "m^3": (1.0, "m^3"),
    "m3": (1.0, "m^3"),
    "cm^3": (1e-6, "m^3"),
    "cm3": (1e-6, "m^3"),
    "L": (1e-3, "m^3"),
    "l": (1e-3, "m^3"),
    "mL": (1e-6, "m^3"),
    "ml": (1e-6, "m^3"),

    # time
    "s": (1.0, "s"),
    "ms": (1e-3, "s"),
    "us": (1e-6, "s"),
    "min": (60.0, "s"),
    "h": (3600.0, "s"),

    # speed
    "m/s": (1.0, "m/s"),
    "km/h": (1000.0 / 3600.0, "m/s"),
    "cm/s": (1e-2, "m/s"),
    "mm/s": (1e-3, "m/s"),

    # acceleration
    "m/s^2": (1.0, "m/s^2"),
    "m/s2": (1.0, "m/s^2"),
    "cm/s^2": (1e-2, "m/s^2"),
    "cm/s2": (1e-2, "m/s^2"),

    # mass
    "kg": (1.0, "kg"),
    "g": (1e-3, "kg"),
    "mg": (1e-6, "kg"),
    "ton": (1e3, "kg"),

    # force
    "N": (1.0, "N"),
    "mN": (1e-3, "N"),
    "uN": (1e-6, "N"),
    "kN": (1e3, "N"),

    # energy
    "J": (1.0, "J"),
    "J/m^3": (1.0, "J/m^3"),
    "mJ": (1e-3, "J"),
    "uJ": (1e-6, "J"),
    "nJ": (1e-9, "J"),
    "kJ": (1e3, "J"),
    "Wh": (3600.0, "J"),
    "kWh": (3.6e6, "J"),
    "eV": (1.602176634e-19, "J"),

    # power
    "W": (1.0, "W"),
    "mW": (1e-3, "W"),
    "kW": (1e3, "W"),
    "MW": (1e6, "W"),

    # charge
    "C": (1.0, "C"),
    "mC": (1e-3, "C"),
    "uC": (1e-6, "C"),
    "nC": (1e-9, "C"),
    "pC": (1e-12, "C"),

    # voltage
    "V": (1.0, "V"),
    "mV": (1e-3, "V"),
    "kV": (1e3, "V"),

    # current
    "A": (1.0, "A"),
    "mA": (1e-3, "A"),
    "uA": (1e-6, "A"),
    "kA": (1e3, "A"),

    # resistance
    "ohm": (1.0, "ohm"),
    "kohm": (1e3, "ohm"),
    "Mohm": (1e6, "ohm"),

    # capacitance
    "F": (1.0, "F"),
    "mF": (1e-3, "F"),
    "uF": (1e-6, "F"),
    "nF": (1e-9, "F"),
    "pF": (1e-12, "F"),

    # inductance
    "H": (1.0, "H"),
    "mH": (1e-3, "H"),
    "uH": (1e-6, "H"),

    # frequency / angular frequency
    "Hz": (1.0, "Hz"),
    "kHz": (1e3, "Hz"),
    "MHz": (1e6, "Hz"),
    "GHz": (1e9, "Hz"),
    "rad/s": (1.0, "rad/s"),
    "rad/s^2": (1.0, "rad/s^2"),

    # magnetic quantities
    "T": (1.0, "T"),
    "mT": (1e-3, "T"),
    "uT": (1e-6, "T"),
    "Wb": (1.0, "Wb"),
    "mWb": (1e-3, "Wb"),
    "uWb": (1e-6, "Wb"),

    # electric field
    "N/C": (1.0, "N/C"),
    "V/m": (1.0, "V/m"),

    # density / pressure
    "kg/m^3": (1.0, "kg/m^3"),
    "kg/m3": (1.0, "kg/m^3"),
    "g/cm^3": (1e3, "kg/m^3"),
    "g/cm3": (1e3, "kg/m^3"),
    "Pa": (1.0, "Pa"),
    "kPa": (1e3, "Pa"),
    "MPa": (1e6, "Pa"),

    # turn density
    "turn/m": (1.0, "turn/m"),
    "turns/m": (1.0, "turn/m"),
    "1/m": (1.0, "1/m"),

    # common constants units, keep unchanged
    "N*m^2/C^2": (1.0, "N*m^2/C^2"),
    "N*m2/C2": (1.0, "N*m^2/C^2"),
    "Nm2/C2": (1.0, "N*m^2/C^2"),
    "F/m": (1.0, "F/m"),
    "H/m": (1.0, "H/m"),
    
    # Combined units
    "N*m": (1.0, "N*m"),
    "kg*m/s": (1.0, "kg*m/s"),
}


def _split_unit_power(unit: str) -> Tuple[str, int]:
    m = re.fullmatch(r"(.+?)\^([-+]?\d+)", unit)
    if m:
        return m.group(1), int(m.group(2))
    m = re.fullmatch(r"([A-Za-z]+)([-+]?\d+)", unit)
    if m:
        return m.group(1), int(m.group(2))
    return unit, 1


def _add_unit_power(powers: Dict[str, int], unit: str, exponent: int) -> None:
    if not unit:
        return
    for part in unit.split("*"):
        if not part:
            continue
        base, power = _split_unit_power(part)
        powers[base] = powers.get(base, 0) + power * exponent
        if powers[base] == 0:
            del powers[base]


def _format_compound_unit(powers: Dict[str, int]) -> str:
    if not powers:
        return ""
    num, den = [], []
    for base, power in powers.items():
        target = num if power > 0 else den
        p = abs(power)
        target.append(base if p == 1 else f"{base}^{p}")
    if not den:
        return "*".join(num)
    return f"{'*'.join(num) or '1'}/{'*'.join(den)}"


def _normalize_compound_unit(cleaned: str) -> Optional[Tuple[float, str]]:
    if not cleaned or not re.search(r"[*/]", cleaned):
        return None

    factor = 1.0
    powers: Dict[str, int] = {}
    sign = 1
    pos = 0
    for match in re.finditer(r"([*/]?)([^*/]+)", cleaned):
        if match.start() != pos:
            return None
        pos = match.end()
        op, token = match.group(1), match.group(2)
        if op == "/":
            sign = -1
        elif op == "*":
            sign = 1

        token = token.strip()
        if not token:
            return None
        token_factor, token_unit = UNIT_TABLE.get(token, (None, None))
        if token_factor is None:
            base, power = _split_unit_power(token)
            if base == token or base not in UNIT_TABLE:
                return None
            base_factor, base_unit = UNIT_TABLE[base]
            token_factor = base_factor ** power
            token_unit = base_unit if power == 1 else f"{base_unit}^{power}"

        if sign > 0:
            factor *= float(token_factor)
        else:
            factor /= float(token_factor)
        _add_unit_power(powers, str(token_unit), sign)

    if pos != len(cleaned):
        return None
    return factor, _format_compound_unit(powers)


def normalize_unit(unit: Any) -> Tuple[float, str, str]:
    """
    Return:
        factor_to_SI, si_unit, status
    """
    cleaned = clean_unit(unit)
    if cleaned in UNIT_TABLE:
        factor, si_unit = UNIT_TABLE[cleaned]
        return factor, si_unit, "ok"
    compound = _normalize_compound_unit(cleaned)
    if compound is not None:
        factor, si_unit = compound
        return factor, si_unit, "ok"
    return 1.0, cleaned, "unknown_unit"


def clean_float_artifact(value: float, significant_digits: int = 15) -> float:
    """Trim binary floating-point artifacts while keeping a numeric float."""
    if not math.isfinite(value):
        return value
    return float(f"{value:.{significant_digits}g}")


def normalize_quantity(value: Any, unit: Any) -> Dict[str, Any]:
    """
    Normalize one quantity to SI.

    If value is symbolic:
    - keep value unchanged
    - still normalize unit if possible
    """
    original_value = value
    original_unit = "" if unit is None else str(unit)

    numeric_value, is_numeric = parse_numeric_value(value)
    factor, si_unit, unit_status = normalize_unit(unit)

    if is_numeric:
        si_value = clean_float_artifact(numeric_value * factor)
        return {
            "value": si_value,
            "unit": si_unit,
            "original_value": original_value,
            "original_unit": original_unit,
            "is_numeric": True,
            "status": unit_status,
        }

    return {
        "value": original_value,
        "unit": si_unit,
        "original_value": original_value,
        "original_unit": original_unit,
        "is_numeric": False,
        "status": "symbolic_value" if unit_status == "ok" else unit_status,
    }


def process_module3(module2_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Module 2 output to SI.

    This function intentionally does NOT:
    - infer formulas
    - infer geometry
    - add physics hints
    - add constants
    - modify conditions semantically
    """
    if not isinstance(module2_json, dict):
        raise TypeError("module2_json must be a dictionary")

    raw_givens = module2_json.get("given", [])
    raw_conditions = module2_json.get("conditions", [])
    raw_targets = module2_json.get("target", [])

    if not isinstance(raw_givens, list):
        raw_givens = []
    if not isinstance(raw_conditions, list):
        raw_conditions = []
    if not isinstance(raw_targets, list):
        raw_targets = []

    normalized_givens: List[Dict[str, Any]] = []

    for item in raw_givens:
        if not isinstance(item, dict):
            continue

        norm = normalize_quantity(item.get("value"), item.get("unit", ""))

        normalized_item = {
            "name": item.get("name", ""),
            "symbol": item.get("symbol", ""),
            "value": norm["value"],
            "unit": norm["unit"],
            "original_value": norm["original_value"],
            "original_unit": norm["original_unit"],
            "is_numeric": norm["is_numeric"],
            "status": norm["status"],
        }

        # Preserve optional metadata from Module 2 without overriding normalized fields.
        for key, val in item.items():
            if key not in normalized_item and key not in {"value", "unit"}:
                normalized_item[key] = val

        normalized_givens.append(normalized_item)

    return {
        "given": normalized_givens,
        "conditions": copy.deepcopy(raw_conditions),
        "target": copy.deepcopy(raw_targets),
    }


# Backward-compatible aliases.
def normalize_to_si(module2_json: Dict[str, Any]) -> Dict[str, Any]:
    return process_module3(module2_json)


def normalize_value_unit_pair(value: Any, unit: Any) -> Tuple[Any, str]:
    out = normalize_quantity(value, unit)
