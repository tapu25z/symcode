#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse và verify answer so với gold label.

Verifier chịu trách nhiệm parse số/unit từ LaTeX, chuẩn hóa đơn vị về SI,
so sánh text answer, và xuất lý do fail rõ ràng cho report.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp

from .module3_units import canonicalize_unit_text, normalize_quantity

_SUPERSCRIPT_DIGITS = str.maketrans({
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

_UNIT_ALTERNATIVES = (
    r"N\*m\^?2/C\^?2|kg\*m/s\^?2|kg\*m/s|kg/m\^?3|J/m\^?3|"
    r"m/s\^?2|m/s²|cm/s\^?2|cm/s²|km/h|m/s|N/C|V/m|"
    r"km\^?2|cm\^?2|mm\^?2|m\^?2|km\^?3|cm\^?3|mm\^?3|m\^?3|"
    r"mC|uC|nC|pC|C|mWb|uWb|Wb|kohm|Mohm|ohm|Ω|Omega|"
    r"degC|°C|kPa|MPa|Pa|cm|mm|km|kg|mg|g|ms|us|min|Hz|kHz|MHz|GHz|"
    r"kWh|J\*s|N\*m|mJ|uJ|nJ|kJ|J|mN|uN|kN|N|mV|kV|V|mA|uA|kA|A|"
    r"mF|uF|nF|pF|F|mH|uH|H|mT|uT|T|rad/s\^?2|rad/s²|rad/s|rad|deg|m|s|h|W|%"
)
_UNIT_TOKEN_PATTERN = re.compile(
    rf"(?<![A-Za-z])(?:{_UNIT_ALTERNATIVES})(?![A-Za-z])"
)


def _strip_simple_boxed(text: str) -> str:
    s = str(text or "").strip()
    if s.startswith(r"\boxed{") and s.endswith("}"):
        return s[len(r"\boxed{"):-1].strip()
    return s


def _replace_simple_latex_fractions(s: str) -> str:
    for _ in range(8):
        new = re.sub(
            r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}",
            r"((\1)/(\2))",
            s,
        )
        if new == s:
            break
        s = new
    return s


def _replace_simple_latex_roots(s: str) -> str:
    for _ in range(8):
        new = re.sub(r"\\sqrt\s*\{\s*([^{}]+?)\s*\}", r"sqrt(\1)", s)
        if new == s:
            break
        s = new
    return s


def _normalize_unit_power_suffix(power: str) -> str:
    if not power:
        return ""
    p = power.strip().translate(_SUPERSCRIPT_DIGITS)
    if p in {"2", "3"}:
        return "^" + p
    m = re.fullmatch(r"\^\s*(?:\{\s*([-+]?\d+)\s*\}|([-+]?\d+))", p)
    return "^" + (m.group(1) or m.group(2)) if m else p


def _latex_unit_fragment_to_text(unit: str) -> str:
    s = _strip_simple_boxed(str(unit or "").strip()).translate(_SUPERSCRIPT_DIGITS)
    if not s:
        return ""
    if "=" in s:
        s = s.split("=", 1)[1]

    replacements = {
        "\\left": "",
        "\\right": "",
        "\\,": " ",
        "\\;": " ",
        "\\:": " ",
        "\\!": " ",
        "\\ ": " ",
        "\\cdot": "*",
        "\\times": "*",
        "\\over": "/",
        "\\Omega": "ohm",
        "\\ohm": "ohm",
        "\\%": "%",
        "\\mu": "u",
        "×": "*",
        "·": "*",
        "⋅": "*",
        "Ω": "ohm",
        "Ω": "ohm",
        "μ": "u",
        "µ": "u",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    for _ in range(5):
        new = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", r"\1/\2", s)
        if new == s:
            break
        s = new

    block_pattern = re.compile(
        r"\\(?:mathrm|text|mathbf)\s*\{\s*([^{}]+?)\s*\}"
        r"(\s*\^\s*(?:\{\s*[-+]?\d+\s*\}|[-+]?\d+)|[23])?"
    )
    for _ in range(5):
        new = block_pattern.sub(
            lambda m: m.group(1).strip() + _normalize_unit_power_suffix(m.group(2) or ""),
            s,
        )
        if new == s:
            break
        s = new

    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\((?:approx|xấp xỉ|gần bằng).*?\)", " ", s, flags=re.I)
    s = re.sub(r"\b(?:approx|xấp xỉ|gần bằng)\b.*$", " ", s, flags=re.I)
    s = re.sub(r"[\[\](),;.!?]+", " ", s)
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s*\*\s*", "*", s)
    s = re.sub(r"\s+", "*", s.strip())
    s = re.sub(r"\*+", "*", s).strip("*")
    s = re.sub(r"\*([A-Za-z]+)\^-([0-9]+)", r"/\1^\2", s)
    s = re.sub(r"(?<![A-Za-z0-9])([A-Za-z]+)\^-([0-9]+)", r"1/\1^\2", s)
    return s


def _split_leading_value_and_unit_tail(text: str) -> Tuple[str, str]:
    s = _strip_simple_boxed(str(text or "").strip()).translate(_SUPERSCRIPT_DIGITS)
    if "=" in s:
        s = s.split("=", 1)[1].strip()

    numeric_patterns = [
        r"\s*\\frac\s*\{\s*[-+]?\d+(?:\.\d+)?\s*\}\s*\{\s*[-+]?\d+(?:\.\d+)?\s*\}",
        r"\s*[-+]?\d+(?:\.\d+)?\s*(?:\\times|×|x|\*)\s*10\s*(?:\^\s*\{?\s*[-+]?\d+\s*\}?|[-+]?\d+)",
        r"\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
    ]
    for pattern in numeric_patterns:
        m = re.match(pattern, s, flags=re.I)
        if m:
            return s[:m.end()], s[m.end():]
    return "", s


def _unit_from_fragment(unit: str) -> str:
    cleaned = _latex_unit_fragment_to_text(unit)
    if not cleaned:
        return ""
    unit = _canonical_known_unit(cleaned)
    if unit:
        return unit
    compact = cleaned.replace("*", "")
    return _canonical_known_unit(compact) if compact != cleaned else ""


def _unit_tail_after_leading_value(text: str) -> str:
    _, tail = _split_leading_value_and_unit_tail(text)
    return _unit_from_fragment(tail)


def _strip_unit_tail_from_expression(text: str) -> str:
    value_part, tail = _split_leading_value_and_unit_tail(text)
    return value_part if value_part and _unit_from_fragment(tail) else text


def _latex_to_expression_text(text: Any, strip_units: bool = True) -> str:
    """
    Convert one answer segment into a single arithmetic expression.
    """
    s = _strip_simple_boxed(str(text or "").strip()).translate(_SUPERSCRIPT_DIGITS)
    if not s:
        return ""

    if "=" in s:
        s = s.split("=", 1)[1]

    s = s.replace("\\left", "").replace("\\right", "")
    if strip_units:
        s = _strip_unit_tail_from_expression(s)
    s = _replace_simple_latex_fractions(s)
    s = _replace_simple_latex_roots(s)

    if strip_units:
        s = re.sub(r"(?:\\mu|μ|µ)\s*\\(?:mathrm|text|mathbf)\s*\{[^{}]*?\}(?:\s*\^\s*\{?\s*[-+]?\d+\s*\}?)?", " ", s)
        s = re.sub(r"\\(?:mathrm|text|mathbf)\s*\{[^{}]*?\}(?:\s*\^\s*\{?\s*[-+]?\d+\s*\}?)?", " ", s)
        s = s.replace("\\Omega", " ").replace("\\ohm", " ")
        s = s.replace("\\%", " ").replace("%", " ")
        s = s.replace("Ω", " ").replace("Ω", " ")
        s = re.sub(rf"(?<=\d)\s*(?:{_UNIT_ALTERNATIVES})\s*$", " ", s)
    else:
        for _ in range(5):
            s = re.sub(r"\\(?:mathrm|text|mathbf)\s*\{\s*([^{}]+?)\s*\}", r"\1", s)

    replacements = {
        "\\times": "*",
        "\\cdot": "*",
        "\\,": " ",
        "\\;": " ",
        "\\:": " ",
        "\\!": " ",
        "\\ ": " ",
        "×": "*",
        "·": "*",
        "⋅": "*",
        "π": "pi",
        "\\pi": "pi",
        "−": "-",
        "–": "-",
        "—": "-",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    s = re.sub(r"\^\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}", r"**(\1)", s)
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace(",", ".")

    s = re.sub(r"(?<=\d)\s*(?=sqrt\s*\()", "*", s)
    s = re.sub(r"(?<=\d)\s*(?=pi\b)", "*", s)
    s = re.sub(r"(?<=\d)\s*(?=\()", "*", s)
    s = re.sub(r"(?<=\))\s*(?=(?:\d|sqrt\s*\(|pi\b|\())", "*", s)

    return re.sub(r"\s+", "", s).strip()


def _has_unresolved_symbols(expr_text: str) -> bool:
    tmp = re.sub(r"(?<=\d)[eE][-+]?\d+", "", expr_text)
    tmp = re.sub(r"\bsqrt\b|\bpi\b", "", tmp)
    tmp = re.sub(r"\b(?:g|e|k|c|h|G|R)\b", "", tmp)
    return bool(re.search(r"[A-Za-z_]", tmp))


def parse_numeric_expression(text: Optional[str]) -> Optional[float]:
    """Evaluate a single numeric expression after stripping units."""
    if not text:
        return None

    expr_text = _latex_to_expression_text(text, strip_units=True)
    if not expr_text or _has_unresolved_symbols(expr_text):
        return None

    try:
        expr = sp.sympify(expr_text, locals={"sqrt": sp.sqrt, "pi": sp.pi})
        if getattr(expr, "free_symbols", None):
            return None
        value = float(expr.evalf())
    except Exception:
        return None

    return value if math.isfinite(value) else None


def _canonical_known_unit(unit: str) -> str:
    canonical = normalize_latex_unit(unit)
    if not canonical:
        return ""
    from .module3_units import UNIT_TABLE
    if canonical in UNIT_TABLE:
        return canonical
    if _UNIT_TOKEN_PATTERN.fullmatch(canonical):
        return canonical
    if re.fullmatch(rf"(?:{_UNIT_ALTERNATIVES})(?:[*/](?:{_UNIT_ALTERNATIVES}))+", canonical):
        return canonical
    return ""



def latex_to_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None

    raw = str(text).strip()
    raw = raw.translate(_SUPERSCRIPT_DIGITS)
    if ";" in raw:
        first_part = next((p.strip() for p in raw.split(";") if p.strip()), "")
        return latex_to_float(first_part) if first_part else None

    expression_value = parse_numeric_expression(raw)
    if expression_value is not None:
        return expression_value

    raw_no_unit = raw
    for _ in range(5):
        raw_no_unit = re.sub(r"\\(?:mathrm|text|mathbf)\s*\{[^{}]*\}", " ", raw_no_unit)
    raw_no_unit = raw_no_unit.strip()

    plain_frac = re.fullmatch(
        r"\s*([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)\s*",
        raw_no_unit,
    )
    if plain_frac:
        a, b = float(plain_frac.group(1)), float(plain_frac.group(2))
        if b != 0: return a / b

    frac = re.search(
        r"\\frac\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}",
        raw,
    )
    if frac:
        a, b = float(frac.group(1)), float(frac.group(2))
        if b != 0: return a / b

    sci = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*(?:\\times|×|x|\*)\s*10\s*(?:\^\s*\{?\s*([-+]?\d+)\s*\}?|([-+]?\d+))",
        raw,
        flags=re.IGNORECASE,
    )
    if sci:
        exponent = sci.group(2) or sci.group(3)
        return float(sci.group(1)) * (10 ** int(exponent))

    pi_match = re.search(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\\pi", raw)
    if pi_match:
        return float(pi_match.group(1)) * math.pi

    s = raw_no_unit.replace("\\,", " ").replace("\\;", " ").replace("\\ ", " ")
    s = s.replace("{", " ").replace("}", " ").replace("×", "*")
    num = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if num:
        try: return float(num.group(0))
        except: pass
    return None


def normalize_latex_unit(unit: str) -> str:
    return canonicalize_unit_text(unit, collapse_electric_field=False)


def to_submission_unit(unit: str) -> str:
    parts = split_multi_value(unit)
    if len(parts) > 1:
        canonical_parts = []
        for part in parts:
            canonical = to_submission_unit(part)
            canonical_parts.append(canonical if (canonical and canonical != "N/A") else "N/A")
        return "; ".join(canonical_parts)
    u = canonicalize_unit_text(unit, collapse_electric_field=True)
    if u == "E_FIELD":
        return "V/m"
    return u if u else "N/A"


def extract_unit_from_boxed(boxed: str) -> str:
    if not boxed: return ""
    s = boxed
    if "=" in s: s = s.split("=", 1)[1]
    # print(f"DEBUG: extract_unit_from_boxed input='{boxed}' -> s='{s}'")
    
    s = s.replace("\\,", " ").replace("\\ ", " ")
    s = s.replace("\\Omega", "Ω").replace("\\%", "%")
    s = s.replace("μ", "u").replace("µ", "u")

    unit = _unit_tail_after_leading_value(s)
    if unit: return unit
    
    # 1. Micro
    micro = re.search(r"(?:\\mu|u|μ|µ)\s*(?:\\(?:mathrm|text|mathbf)\s*\{\s*([A-Za-zΩ/%^0-9*]+)\s*\}|([A-Za-zΩ/%^0-9*]+))", s, re.I)
    if micro:
        unit = _canonical_known_unit("u" + (micro.group(1) or micro.group(2)).strip())
        if unit: return unit

    # 2. LaTeX blocks
    # Nếu không có số, thử parse toàn bộ chuỗi như một unit fragment.
    if not re.search(r"\d", s):
        unit = _unit_from_fragment(s)
        if unit: return unit

    # Parse individual blocks
    for match in re.finditer(r"\\(?:mathrm|text|mathbf)\s*\{\s*([^{}]+?)\s*\}", s):
        val = match.group(1).strip()
        unit = _canonical_known_unit(val)
        if unit: return unit

    if "Ω" in s: return normalize_latex_unit("Ω")
    if "%" in s: return "%"

    # 3. Heuristic
    clean = re.sub(r"\\(?:mathrm|text|mathbf)\s*\{([^}]+?)\}", r" \1 ", s)
    clean = clean.replace("{", " ").replace("}", " ").strip()
    clean = re.sub(r"[\.\s,;!\?]+$", "", clean)
    clean = re.sub(r"\(?(?:approx|xấp xỉ|gần bằng).*?\)?$", "", clean, re.I).strip()

    for match in reversed(list(_UNIT_TOKEN_PATTERN.finditer(clean))):
        if match.start() > 0:
            if not clean[match.start()-1].isalpha():
                unit = _canonical_known_unit(match.group(0))
                if unit: return unit
        else:
            unit = _canonical_known_unit(match.group(0))
            if unit: return unit
    return ""


def extract_numeric_values(text: str) -> List[float]:
    if not text: return []
    raw = str(text).translate(_SUPERSCRIPT_DIGITS)
    spans: List[Tuple[int, int, float]] = []

    def add_matches(pattern: str, converter, source: str = raw, flags: int = 0) -> str:
        masked = list(source)
        for m in re.finditer(pattern, source, flags=flags):
            try: spans.append((m.start(), m.end(), float(converter(m))))
            except: continue
            for idx in range(m.start(), m.end()): masked[idx] = " "
        return "".join(masked)

    working = add_matches(r"\\frac\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}", lambda m: float(m.group(1))/float(m.group(2)))
    working = add_matches(r"(?<![\d.])([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)(?![\d.])", lambda m: float(m.group(1))/float(m.group(2)), working)
    working = add_matches(r"([-+]?\d+(?:\.\d+)?)\s*(?:\\times|×|x|\*)\s*10\s*(?:\^\s*\{?\s*([-+]?\d+)\s*\}?|([-+]?\d+))", lambda m: float(m.group(1))*(10**int(m.group(2) or m.group(3))), working, re.I)

    working = re.sub(r"\\(?:mathrm|text)\s*\{[^}]*\}", " ", working)
    working = working.replace("\\,", " ").replace("\\;", " ").replace("\\ ", " ").replace("{", " ").replace("}", " ")
    for m in re.finditer(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", working):
        spans.append((m.start(), m.end(), float(m.group(0))))
    return [v for _, _, v in sorted(spans, key=lambda x: x[0])]


def extract_units_from_segment(text: str) -> List[str]:
    if not text: return []
    raw = str(text)
    whole_unit = _unit_tail_after_leading_value(raw)
    if whole_unit:
        return [whole_unit]
    if not re.search(r"\d", raw):
        whole_unit = _unit_from_fragment(raw)
        if whole_unit:
            return [whole_unit]
    units = []
    for m in re.finditer(r"(?:\\mu|μ|µ)\s*\\(?:mathrm|text|mathbf)\s*\{\s*([^}]+?)\s*\}", raw):
        u = _canonical_known_unit("u" + m.group(1))
        if u: units.append(u)
    for m in re.finditer(r"\\(?:mathrm|text|mathbf)\s*\{\s*([^}]+?)\s*\}", raw):
        u = _canonical_known_unit(m.group(1))
        if u: units.append(u)
    if "\\Omega" in raw: units.append(normalize_latex_unit("\\Omega"))
    if "\\%" in raw or "%" in raw: units.append("%")
    clean = re.sub(r"\\(?:mathrm|text|mathbf)\s*\{[^}]*\}", " ", raw).replace("\\Omega", "Ω").replace("\\%", "%").replace("μ", "u").replace("µ", "u")
    for m in _UNIT_TOKEN_PATTERN.finditer(clean):
        u = normalize_latex_unit(m.group(0))
        if u: units.append(u)
    res = []
    for u in units:
        if u not in res: res.append(u)
    return res


def extract_number_unit_pairs(boxed: str) -> List[Tuple[float, str]]:
    if not boxed: return []
    parts = [p.strip() for p in re.split(r"\s*;\s*", boxed) if p.strip()] or [boxed.strip()]
    pairs = []
    for p in parts:
        v = parse_numeric_expression(p)
        if v is None:
            v = latex_to_float(p)
        if v is None: continue
        us = extract_units_from_segment(p) or [extract_unit_from_boxed(p)]
        pairs.append((v, us[0] if us else ""))
    return pairs


def split_multi_value(text: str) -> List[str]:
    raw = str(text or "").strip()
    return [p.strip() for p in re.split(r"\s*;\s*", raw) if p.strip()]


def normalize_text_answer(x: Any) -> str:
    s = str(x or "").strip().replace("\\boxed{", "").strip("{} ").lower()
    s = re.sub(r"\s+", " ", s)
    mapping = {"yes": "yes", "y": "yes", "no": "no", "n": "no", "true": "true", "false": "false", "đúng": "true", "sai": "false"}
    return mapping.get(s, s)


def normalize_choice_text(x: Any) -> str:
    s = normalize_text_answer(x)
    s = re.sub(r"^(?:the\s+)?(?:circuit\s+)?(?:exhibits?|has|is)\s+", "", s)
    s = re.sub(r"\bcharacteristic\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _get_module2_options(record: Dict[str, Any]) -> Dict[str, str]:
    m2 = record.get("module2_extract") or record.get("module2") or {}
    targets = m2.get("target", []) if isinstance(m2, dict) else []
    if targets and isinstance(targets[0], dict):
        opts = targets[0].get("options")
        if isinstance(opts, dict): return {str(k).strip().lower(): str(v).strip() for k, v in opts.items()}
    return {}


def _get_module2_target_unit(record: Dict[str, Any]) -> str:
    m2 = record.get("module2_extract") or record.get("module2") or {}
    targets = m2.get("target", []) if isinstance(m2, dict) else []
    if targets and isinstance(targets[0], dict):
        return str(targets[0].get("unit", "") or "").strip()
    return ""


def _prediction_target_unit(record: Dict[str, Any], gold_unit: str) -> str:
    target_unit = _get_module2_target_unit(record)
    if target_unit and target_unit.strip().lower() not in {"unit", "units", "n/a", "na"} and normalize_latex_unit(target_unit):
        return target_unit
    return gold_unit


def _choice_tokens(answer: str, options: Dict[str, str]) -> set[str]:
    norm = normalize_choice_text(answer)
    tokens = set()
    if norm in options:
        tokens.add(norm)
        tokens.add(normalize_choice_text(options[norm]))
        return {t for t in tokens if t}
    for l in re.findall(r"(?<![a-z0-9])[a-d](?![a-z0-9])", norm):
        if l in options:
            tokens.add(l)
            tokens.add(normalize_choice_text(options[l]))
    for l, t in options.items():
        ot = normalize_choice_text(t)
        if ot and (norm == ot or (len(norm) >= 4 and len(ot) >= 4 and (ot in norm or norm in ot))):
            tokens.add(l); tokens.add(ot)
    if not tokens and norm:
        for p in re.split(r"\s*(?:,|;|/|\\band\\b|\\bor\\b)\s*", norm):
            if p.strip(): tokens.add(p.strip())
    return {t for t in tokens if t}


def normalize_symbolic_answer(x: Any) -> str:
    s = str(x or "").strip().replace("\\boxed{", "").strip("{} ").lower().replace("\\\\", "\\")
    s = re.sub(r"\\(?:mathrm|text)\s*\{[^}]*\}", "", s)
    s = re.sub(r"\\sqrt\s*\{\s*([^}]+?)\s*\}", r"sqrt(\1)", s)
    s = s.replace("\\", "").replace(".", "*").replace(" ", "").replace("{", "").replace("}", "")
    return s


def is_text_answer_type(answer_type: str, gold_answer: str) -> bool:
    g = normalize_text_answer(gold_answer)
    if answer_type == "multiple_choice": return latex_to_float(gold_answer) is None
    if answer_type in {"yes_no", "conceptual_qa"}: return True
    return g in {"yes", "no", "true", "false", "a", "b", "c", "d"}


def verify_yes_no_text(gold_answer: str, boxed: str) -> Dict[str, Any]:
    gn, pn = normalize_text_answer(gold_answer), normalize_text_answer(boxed)
    return {"status": "pass" if gn == pn else "fail", "reason": "ok" if gn == pn else "answer_mismatch", "gold_text": gn, "pred_text": pn}


def verify_multiple_choice_text(record: Dict[str, Any], gold_answer: str, boxed: str) -> Dict[str, Any]:
    opts = _get_module2_options(record)
    gt, pt = _choice_tokens(gold_answer, opts), _choice_tokens(boxed, opts)
    ok = bool(gt & pt) if opts and gt and pt else normalize_choice_text(gold_answer) == normalize_choice_text(boxed)
    return {"status": "pass" if ok else "fail", "reason": "ok" if ok else "answer_mismatch", "gold_text": normalize_text_answer(gold_answer), "pred_text": normalize_text_answer(boxed)}


def units_equivalent(unit_a: str, unit_b: str) -> bool:
    a, b = canonicalize_unit_text(unit_a, True), canonicalize_unit_text(unit_b, True)
    if a == b: return True
    try:
        na, nb = normalize_quantity(1, a), normalize_quantity(1, b)
        if na["status"] != "unknown_unit" and nb["status"] != "unknown_unit": return na["unit"] == nb["unit"]
    except: pass
    return False


def convert_value_to_si(value: float, unit: str) -> Tuple[float, str]:
    norm = normalize_quantity(value, unit)
    if norm.get("is_numeric") and norm.get("status") != "unknown_unit": return float(norm["value"]), str(norm["unit"])
    return float(value), normalize_latex_unit(unit)


def gold_rounding_abs_tolerance_si(gold_answer: str, gold_unit: str) -> float:
    raw = str(gold_answer or "").strip().replace(",", ".").translate(_SUPERSCRIPT_DIGITS)
    m = re.fullmatch(r"[-+]?(?:(\d+)(?:\.(\d+))?|\.(\d+))(?:[eE]([-+]?\d+))?", raw)
    if not m: return 0.0
    decimals = m.group(2) or m.group(3) or ""
    raw_tol = 0.5 * (10 ** (int(m.group(4) or 0) - len(decimals)))
    if not math.isfinite(raw_tol) or raw_tol <= 0: return 0.0
    val_si, _ = convert_value_to_si(raw_tol, gold_unit)
    return abs(val_si)


def numeric_within_tolerance(pred_si: float, gold_si: float, gold_answer: str, gold_unit: str) -> Tuple[bool, float, float, float]:
    ae = abs(pred_si - gold_si)
    re_err = ae / max(abs(gold_si), 1e-12)
    rt = gold_rounding_abs_tolerance_si(gold_answer, gold_unit)
    ok = ae <= 1e-6 or re_err <= 1e-2 or (rt > 0 and ae <= rt)
    return ok, ae, re_err, rt


def choose_prediction_value_for_gold(boxed: str, fallback_v: Optional[float], fallback_u: str, gold_u: str) -> Tuple[Optional[float], str]:
    pairs = extract_number_unit_pairs(boxed)
    if fallback_u:
        pairs = [(v, fallback_u) for v, _ in pairs]
    if gold_u and pairs:
        for v, u in pairs:
            if u and units_equivalent(u, gold_u): return v, u
    return pairs[0] if pairs else (fallback_v, fallback_u)


def _numeric_prediction_text(execution: Dict[str, Any]) -> str:
    boxed = str(execution.get("boxed", "") or "").strip()
    stdout = str(execution.get("stdout", "") or "").strip()
    if stdout and (not extract_unit_from_boxed(boxed)) and extract_unit_from_boxed(stdout):
        return stdout
    return boxed


def verify_multi_numeric_against_gold(gold_answer: str, gold_unit: str, boxed: str, fallback_v: Optional[float], fallback_u: str) -> Optional[Dict[str, Any]]:
    gvs, gus = split_multi_value(gold_answer), split_multi_value(gold_unit)
    if len(gvs) <= 1 and len(gus) <= 1: return None
    if len(gus) == 1 and len(gvs) > 1: gus = gus * len(gvs)
    if not gus: gus = [""] * len(gvs)
    pred_pairs = extract_number_unit_pairs(boxed)
    if fallback_u:
        pred_pairs = [(v, fallback_u) for v, _ in pred_pairs]
    pred_pairs = pred_pairs or ([(fallback_v, fallback_u)] if fallback_v is not None else [])
    if len(pred_pairs) < len(gvs): return {"status": "fail", "reason": "missing_multi_prediction"}
    details, used, all_ok = [], set(), True
    for i, gv in enumerate(gvs):
        gu = gus[i] if i < len(gus) else ""
        gn = normalize_quantity(gv, gu)
        if gn["is_numeric"]: gv_si, gu_si = float(gn["value"]), str(gn["unit"])
        else:
            pv = latex_to_float(gv)
            if pv is None: return {"status": "skip", "reason": "non_numeric_multi_gold"}
            gv_si, gu_si = convert_value_to_si(pv, gu)
        found = False
        for j, (pv_raw, pu_raw) in enumerate(pred_pairs):
            if j in used: continue
            pv_si, pu_si = convert_value_to_si(pv_raw, pu_raw)
            ex_u = gu_si or normalize_latex_unit(gu)
            pr_u = pu_si or normalize_latex_unit(pu_raw)
            u_ok = True if not ex_u else bool(pr_u) and units_equivalent(pr_u, ex_u)
            n_ok, ae, re_err, rt = numeric_within_tolerance(pv_si, gv_si, gv, gu)
            if u_ok and n_ok:
                used.add(j); found = True
                details.append({"unit_ok": True, "numeric_ok": True, "abs_error": ae})
                break
        if not found: all_ok = False; details.append({"unit_ok": False, "numeric_ok": False})
    return {"status": "pass" if all_ok else "fail", "reason": "ok" if all_ok else "multi_answer_mismatch", "details": details}


def verify_against_gold(record: Dict[str, Any], execution: Dict[str, Any], answer_type: str) -> Dict[str, Any]:
    ga = str(record.get("answer", record.get("original_answer", ""))).strip()
    gu = str(record.get("unit", record.get("original_unit", ""))).strip()
    boxed = str(execution.get("boxed", "") or "").strip()
    numeric_boxed = _numeric_prediction_text(execution)
    fallback_unit = _prediction_target_unit(record, gu)
    if not ga: return {"status": "skip", "reason": "missing_gold_answer"}
    if not is_text_answer_type(answer_type, ga) and not gu and re.search(r"E[_A-Za-z]|sqrt|\\sqrt", ga) and re.search(r"E[_A-Za-z]|sqrt|\\sqrt", boxed):
        gn, pn = normalize_symbolic_answer(ga), normalize_symbolic_answer(boxed)
        return {"status": "pass" if gn == pn else "fail", "gold_text": gn, "pred_text": pn}
    if answer_type == "conceptual_qa": return {"status": "skip", "reason": "manual_review", "gold_text": ga, "pred_text": boxed}
    if answer_type == "yes_no" or normalize_text_answer(ga) in {"yes", "no", "true", "false"}: return verify_yes_no_text(ga, boxed)
    if answer_type == "multiple_choice": return verify_multiple_choice_text(record, ga, boxed)

    mr = verify_multi_numeric_against_gold(ga, gu, numeric_boxed, execution.get("numeric_answer"), fallback_unit)
    if mr: return mr

    gn = normalize_quantity(ga, gu)
    if not gn["is_numeric"]:
        gv = latex_to_float(ga)
        if gv is None: return {"status": "skip", "reason": "non_numeric_gold"}
        gv_si, gu_si = convert_value_to_si(gv, gu)
    else: gv_si, gu_si = float(gn["value"]), str(gn["unit"])

    pv_raw, pu_raw = choose_prediction_value_for_gold(numeric_boxed, execution.get("numeric_answer"), fallback_unit, gu)
    if pv_raw is None: return {"status": "skip", "reason": "non_numeric_prediction"}
    pv_si, pu_si = convert_value_to_si(pv_raw, pu_raw)

    ex_u = gu_si or normalize_latex_unit(gu)
    pr_u = pu_si or normalize_latex_unit(pu_raw)
    u_ok = True if not ex_u else bool(pr_u) and units_equivalent(pr_u, ex_u)
    n_ok, ae, re_err, rt = numeric_within_tolerance(pv_si, gv_si, ga, gu)
    ok = u_ok and n_ok
    return {"status": "pass" if ok else "fail", "reason": "ok" if ok else ("unit_mismatch" if not u_ok else "numeric_mismatch"), "unit_ok": u_ok, "numeric_ok": n_ok, "abs_error": ae, "rel_error": re_err}
