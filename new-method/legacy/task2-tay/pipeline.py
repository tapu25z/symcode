#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Qwen3-8B Physics SymCode Pipeline

Pipeline:
  JSON question -> Qwen generates Python/SymPy code -> safety check -> execute code -> parse JSON answer -> evaluate

Kaggle install:
  pip install -U "transformers>=4.51.0" accelerate bitsandbytes sympy torch

Kaggle run:
  python clean_symcode_pipeline_input_only.py --input /kaggle/input/.../test_04_numeric_only_corrected_suggested.jsonl --load-in-4bit

Outputs:
  result.json
  submission.json
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# No built-in TEST_DATA/GOLD. This version reads samples and gold labels from --input.

SYMCODE_SYSTEM_PROMPT = r"""You are an expert physics solver and Python/SymPy code generator.

Given one physics problem JSON, return ONLY executable Python code.
Do not write explanations.
Do not output <think> tags.
A Python code fence is allowed, but no other text is allowed.

The code MUST:
1. import json
2. import sympy as sp
3. define all given quantities with correct unit conversions
4. compute the requested final numeric answer
5. set:
   final_answer = ...
   final_unit = "..."
6. print exactly one JSON object:
   print(json.dumps({"answer": final_answer, "unit": final_unit}))

Important rules:
- The printed JSON must have exactly keys "answer" and "unit".
- final_answer must be a Python int or float, not a SymPy object.
- final_unit must be ASCII only.
- Use SI units unless the problem clearly asks for another target unit.
- For lens/image distance problems using cm in the problem, output image distance in cm.
- Convert prefixes correctly: u = micro = 1e-6, m = milli = 1e-3, n = nano = 1e-9, cm = 1e-2, mm = 1e-3.
- Use "ohm" instead of Ω.
- Use "m/s^2" instead of m/s².
- Use "N/C" for electric field.
- Do not use input(), open(), eval(), exec(), os, sys, subprocess, requests, socket, pathlib, shutil.
- Do not read or write files.
- Keep the code short and deterministic.

Example output code:
import json
import sympy as sp

C = sp.Float(47e-6)
U = sp.Integer(12)
E = sp.Rational(1, 2) * C * U**2

final_answer = float(E)
final_unit = "J"
print(json.dumps({"answer": final_answer, "unit": final_unit}))
"""

DEBUG_SYSTEM_PROMPT = r"""You are fixing Python/SymPy code for a physics problem.

Return ONLY corrected executable Python code.
Do not write explanations.
Do not output <think> tags.
A Python code fence is allowed, but no other text is allowed.

The corrected code MUST print exactly one JSON object:
print(json.dumps({"answer": final_answer, "unit": final_unit}))

Do not use input(), open(), eval(), exec(), os, sys, subprocess, requests, socket, pathlib, shutil.
"""


def load_records(input_path: Optional[str]) -> List[Dict[str, Any]]:
    """Load external data only.

    Supports:
      - JSONL: one JSON object per line
      - JSON object: one sample
      - JSON array: list of samples

    This replaces the old built-in TEST_DATA list.
    """
    if not input_path:
        raise ValueError("Missing --input. This input-only version does not contain built-in 25 samples.")

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # JSONL: first non-space char is usually '{' and there are multiple lines.
    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {line_no} must be a JSON object.")
            records.append(obj)
        return records

    data = json.loads(text)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        if not all(isinstance(x, dict) for x in data):
            raise ValueError("Input JSON array must contain JSON objects.")
        return data
    raise ValueError("Input file must be JSONL, a JSON object, or a JSON array.")


def qid_of(sample: Dict[str, Any]) -> str:
    return str(sample.get("query_id") or sample.get("id") or "")


def query_of(sample: Dict[str, Any]) -> str:
    return str(sample.get("query") or sample.get("question") or "")


def normalize_question_text(q: Any) -> str:
    """Normalize the full problem text for the model prompt.

    This is text cleanup only. It does not convert values to SI.
    Example: ``20 μF`` becomes ``20 uF``, not ``20e-6 F``.
    """
    q = str(q or "")
    q = q.replace("\u00a0", " ")
    q = q.replace("\u200b", "")
    q = q.replace("×", "*")
    q = q.replace("·", "*").replace("⋅", "*")
    q = q.replace("−", "-").replace("–", "-").replace("—", "-")
    q = q.replace("±", "+/-")
    q = q.replace("π", "pi")
    q = q.replace("Ω", "ohm").replace("Ω", "ohm")
    q = q.replace("μ", "u").replace("µ", "u")
    q = q.replace("°C", "degC")
    q = q.replace("°", "deg")
    q = q.translate(str.maketrans({
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
        "⁻": "-", "⁺": "+",
        "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
        "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    }))
    return re.sub(r"\s+", " ", q).strip()


def make_model_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Return the JSON shown to Qwen without leaking the gold answer.

    The raw question is preserved, and a cleaned ``normalized_query`` field is
    added so the model can read Unicode units/symbols more reliably.
    """
    banned_keys = {
        "expected_answer", "gold_answer", "gold", "answer", "correct",
        "reason", "prediction", "symcode", "execution", "attempts",
    }
    out = {k: v for k, v in sample.items() if k not in banned_keys}
    raw_q = query_of(sample)
    if raw_q:
        out["normalized_query"] = normalize_question_text(raw_q)
    return out


def _safe_eval_number_expr(expr: str) -> Optional[float]:
    """Safely evaluate a small numeric expression such as 1/2 or 2*10**-3."""
    allowed_binops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a ** b,
    }
    allowed_unary = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Num):  # pragma: no cover; compatibility with older Python
            return float(node.n)
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binops:
            return allowed_binops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
            return allowed_unary[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported numeric expression")

    try:
        value = float(_eval(ast.parse(expr, mode="eval")))
        return value if math.isfinite(value) else None
    except Exception:
        return None


def parse_numeric_value(x: Any) -> Optional[float]:
    """Parse numeric values found in gold labels or predictions.

    Supported examples: 3, "3", "2,5", "1/2", "6e-8",
    "6 × 10^-8", "6*10⁻⁸". Symbolic values return None.
    """
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        value = float(x)
        return value if math.isfinite(value) else None
    if not isinstance(x, str):
        return None

    s = x.strip()
    if not s:
        return None

    s = s.translate(str.maketrans({
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
        "⁻": "-", "⁺": "+",
    }))
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("×", "*").replace("x", "*").replace("X", "*")
    s = s.replace("·", "*").replace("⋅", "*")
    s = s.replace("^", "**")
    s = re.sub(r"\s+", "", s)

    # Thousands comma: 1,234.5. Decimal comma: 2,5.
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    elif re.fullmatch(r"[-+]?\d+,\d+", s):
        s = s.replace(",", ".")

    # Keep symbolic expressions symbolic, but allow e/E scientific notation.
    tmp = re.sub(r"[eE][-+]?\d+", "", s)
    if re.search(r"[A-Za-zα-ωΑ-Ω]", tmp):
        return None

    s = re.sub(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\*10(?:\*\*)?([-+]?\d+)",
        r"\1e\2",
        s,
        flags=re.IGNORECASE,
    )

    if not re.fullmatch(r"[-+0-9.eE*/()+]+", s):
        return None

    value = _safe_eval_number_expr(s)
    if value is not None:
        return value
    try:
        value = float(s)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def gold_of(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build one gold entry from a record in the external input file."""
    answer = None
    for key in ("expected_answer", "gold_answer", "answer"):
        if key in sample and sample.get(key) not in (None, ""):
            answer = sample.get(key)
            break
    if answer is None:
        return None
    unit = sample.get("unit") or sample.get("expected_unit") or sample.get("gold_unit") or ""
    return {"answer": answer, "unit": unit}


def build_gold_map(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    gold_map: Dict[str, Dict[str, Any]] = {}
    for sample in records:
        qid = qid_of(sample)
        gold = gold_of(sample)
        if qid and gold is not None:
            gold_map[qid] = gold
    return gold_map


def build_symcode_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    payload = json.dumps(make_model_sample(sample), ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": SYMCODE_SYSTEM_PROMPT},
        {"role": "user", "content": "Input JSON:\n" + payload + "\n\nReturn executable Python code only."},
    ]


def build_debug_messages(sample: Dict[str, Any], bad_code: str, execution: Dict[str, Any]) -> List[Dict[str, str]]:
    payload = json.dumps(make_model_sample(sample), ensure_ascii=False, indent=2)
    user_text = f"""Input JSON:
{payload}

Previous code:
```python
{bad_code}
```

Execution status: {execution.get('status')}
Execution stage: {execution.get('stage')}
Error: {execution.get('error')}

Stdout:
{execution.get('stdout', '')}

Stderr:
{execution.get('stderr', '')}

Return corrected executable Python code only."""
    return [
        {"role": "system", "content": DEBUG_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


class QwenSymCodeClient:
    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = bnb_config
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, messages: List[Dict[str, str]]) -> str:
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.temperature and self.temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": self.temperature, "top_p": self.top_p})
        else:
            gen_kwargs.update({"do_sample": False})

        output_ids = self.model.generate(**inputs, **gen_kwargs)[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def remove_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text.strip()


def extract_python_code(text: str) -> str:
    text = remove_thinking(text)
    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    text = re.sub(r"^```(?:python|py)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    return text


BANNED_NAME_CALLS = {"open", "eval", "exec", "input", "compile", "__import__", "globals", "locals", "vars", "dir", "help", "breakpoint"}
BANNED_IMPORT_ROOTS = {"os", "sys", "subprocess", "socket", "requests", "pathlib", "shutil", "http", "urllib", "ftplib", "pickle", "marshal", "builtins"}
ALLOWED_IMPORT_ROOTS = {"json", "math", "sympy"}


def safety_check(code_text: str) -> Optional[str]:
    if not code_text or not code_text.strip():
        return "empty_code"
    banned_patterns = [
        r"\bopen\s*\(", r"\beval\s*\(", r"\bexec\s*\(", r"\binput\s*\(", r"__import__", r"\bcompile\s*\(",
        r"\bsubprocess\b", r"\bsocket\b", r"\brequests\b", r"\bpathlib\b", r"\bshutil\b", r"\bpickle\b", r"\bmarshal\b", r"\bos\.", r"\bsys\.",
    ]
    for pattern in banned_patterns:
        if re.search(pattern, code_text):
            return f"banned_pattern:{pattern}"
    try:
        tree = ast.parse(code_text)
    except SyntaxError as exc:
        return f"syntax_error:{exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BANNED_IMPORT_ROOTS or root not in ALLOWED_IMPORT_ROOTS:
                    return f"banned_import:{alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BANNED_IMPORT_ROOTS or root not in ALLOWED_IMPORT_ROOTS:
                return f"banned_import_from:{node.module}"
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in BANNED_NAME_CALLS:
                return f"banned_call:{fn.id}"
            if isinstance(fn, ast.Attribute) and fn.attr in BANNED_NAME_CALLS:
                return f"banned_attr_call:{fn.attr}"
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            return "banned_with_statement"
    return None


def to_float(x: Any) -> Optional[float]:
    return parse_numeric_value(x)


def parse_prediction_from_stdout(stdout: str) -> Optional[Dict[str, Any]]:
    text = stdout.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def validate_prediction(obj: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    if obj is None:
        return False, "cannot_parse_stdout_json"
    if set(obj.keys()) != {"answer", "unit"}:
        return False, f"wrong_keys:{list(obj.keys())}"
    if to_float(obj.get("answer")) is None:
        return False, "answer_not_numeric"
    try:
        str(obj.get("unit", "")).encode("ascii")
    except UnicodeEncodeError:
        return False, "unit_not_ascii"
    return True, "ok"


def execute_symcode(code_text: str, timeout: int) -> Dict[str, Any]:
    safety_error = safety_check(code_text)
    if safety_error:
        return {"status": "fail", "stage": "safety", "error": safety_error, "stdout": "", "stderr": "", "prediction": None}

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "generated_solution.py"
        script_path.write_text(code_text, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, str(script_path)], cwd=tmpdir, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "fail", "stage": "execution", "error": "timeout", "stdout": "", "stderr": "", "prediction": None}

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        return {"status": "fail", "stage": "execution", "error": f"returncode={proc.returncode}", "stdout": stdout, "stderr": stderr, "prediction": None}

    pred = parse_prediction_from_stdout(stdout)
    ok, reason = validate_prediction(pred)
    if not ok:
        return {"status": "fail", "stage": "parse", "error": reason, "stdout": stdout, "stderr": stderr, "prediction": pred}
    return {"status": "pass", "stage": "done", "error": "", "stdout": stdout, "stderr": stderr, "prediction": pred}


DIMENSIONLESS_UNITS = {
    "", "-", "—", "none", "None", "null",
    "dimensionless", "unitless", "no_unit", "lần", "lan",
}


def canonicalize_unit_text(unit: Any, collapse_electric_field: bool = False) -> str:
    """Clean a unit string into a stable ASCII-like notation.

    This only normalizes spelling/formatting. It does not multiply the value.
    Example: ``μF`` -> ``uF`` and ``m/s²`` -> ``m/s^2``.
    """
    if unit is None:
        return ""

    raw = str(unit).strip()
    if raw in DIMENSIONLESS_UNITS:
        return ""

    u = raw
    u = u.replace("\\cdot", "*").replace("\\times", "*")
    u = u.replace("×", "*").replace("·", "*").replace("⋅", "*")
    u = u.replace("\\,", "").replace("\\;", "").replace("\\:", "").replace("\\!", "").replace("\\ ", "")

    # Remove common LaTeX wrappers: \mathrm{N/C}, \text{ohm}, ...
    for _ in range(5):
        new_u = re.sub(r"\\(?:mathrm|text|mathbf)\s*\{\s*([^{}]+?)\s*\}", r"\1", u)
        if new_u == u:
            break
        u = new_u

    u = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", r"\1/\2", u)
    u = u.replace("\\over", "/")
    u = u.replace("\\Omega", "ohm").replace("\\mu", "u")
    u = u.replace("Ω", "ohm").replace("Ω", "ohm")
    u = u.replace("μ", "u").replace("µ", "u")
    u = u.replace("²", "^2").replace("³", "^3")
    u = u.replace("−", "-").replace("–", "-").replace("—", "-")
    u = re.sub(r"\bper\b", "/", u, flags=re.IGNORECASE)
    u = u.replace("{", "").replace("}", "")
    u = re.sub(r"\s+", "", u)
    u = u.replace(".", "*")

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
        "ohm": "ohm",
        "ohms": "ohm",
        "ω": "ohm",
        "Ω": "ohm",
        "uf": "uF",
        "nf": "nF",
        "pf": "pF",
        "uc": "uC",
        "nc": "nC",
        "pc": "pC",
        "ua": "uA",
        "uh": "uH",
        "m/s²": "m/s^2",
        "m/s2": "m/s^2",
        "cm/s2": "cm/s^2",
        "n/c": "N/C",
        "v/m": "V/m",
        "nc^-1": "N/C",
        "n*c^-1": "N/C",
        "npercalc": "N/C",
        "vperm": "V/m",
        "kg*m/s^2": "N",
        "kg*m/s2": "N",
        "n*m": "N*m",
        "kw*h": "kWh",
        "kwh": "kWh",
        "j*s": "J*s",
    }

    key = u.lower()
    if key in aliases:
        u = aliases[key]

    if collapse_electric_field and u in {"N/C", "V/m"}:
        return "E_FIELD"
    return u


# Unit table maps a canonical unit to: (factor_to_SI, SI_unit).
UNIT_TABLE: Dict[str, Tuple[float, str]] = {
    "": (1.0, ""),
    "%": (0.01, ""),

    # length / area / volume
    "m": (1.0, "m"),
    "km": (1e3, "m"),
    "dm": (1e-1, "m"),
    "cm": (1e-2, "m"),
    "mm": (1e-3, "m"),
    "um": (1e-6, "m"),
    "nm": (1e-9, "m"),
    "m^2": (1.0, "m^2"),
    "m2": (1.0, "m^2"),
    "cm^2": (1e-4, "m^2"),
    "cm2": (1e-4, "m^2"),
    "mm^2": (1e-6, "m^2"),
    "mm2": (1e-6, "m^2"),
    "m^3": (1.0, "m^3"),
    "m3": (1.0, "m^3"),
    "cm^3": (1e-6, "m^3"),
    "cm3": (1e-6, "m^3"),
    "L": (1e-3, "m^3"),
    "mL": (1e-6, "m^3"),

    # time / motion
    "s": (1.0, "s"),
    "ms": (1e-3, "s"),
    "us": (1e-6, "s"),
    "min": (60.0, "s"),
    "h": (3600.0, "s"),
    "m/s": (1.0, "m/s"),
    "km/h": (1000.0 / 3600.0, "m/s"),
    "cm/s": (1e-2, "m/s"),
    "m/s^2": (1.0, "m/s^2"),
    "cm/s^2": (1e-2, "m/s^2"),

    # mass / force / pressure
    "kg": (1.0, "kg"),
    "g": (1e-3, "kg"),
    "mg": (1e-6, "kg"),
    "N": (1.0, "N"),
    "mN": (1e-3, "N"),
    "uN": (1e-6, "N"),
    "kN": (1e3, "N"),
    "Pa": (1.0, "Pa"),
    "kPa": (1e3, "Pa"),
    "MPa": (1e6, "Pa"),

    # energy / power
    "J": (1.0, "J"),
    "mJ": (1e-3, "J"),
    "uJ": (1e-6, "J"),
    "nJ": (1e-9, "J"),
    "kJ": (1e3, "J"),
    "Wh": (3600.0, "J"),
    "kWh": (3.6e6, "J"),
    "W": (1.0, "W"),
    "mW": (1e-3, "W"),
    "kW": (1e3, "W"),

    # electricity / magnetism
    "C": (1.0, "C"),
    "mC": (1e-3, "C"),
    "uC": (1e-6, "C"),
    "nC": (1e-9, "C"),
    "pC": (1e-12, "C"),
    "V": (1.0, "V"),
    "mV": (1e-3, "V"),
    "kV": (1e3, "V"),
    "A": (1.0, "A"),
    "mA": (1e-3, "A"),
    "uA": (1e-6, "A"),
    "ohm": (1.0, "ohm"),
    "kohm": (1e3, "ohm"),
    "Mohm": (1e6, "ohm"),
    "F": (1.0, "F"),
    "mF": (1e-3, "F"),
    "uF": (1e-6, "F"),
    "nF": (1e-9, "F"),
    "pF": (1e-12, "F"),
    "H": (1.0, "H"),
    "mH": (1e-3, "H"),
    "uH": (1e-6, "H"),
    "T": (1.0, "T"),
    "Wb": (1.0, "Wb"),
    "Hz": (1.0, "Hz"),

    # common compound units
    "N/C": (1.0, "N/C"),
    "V/m": (1.0, "N/C"),
    "N*m": (1.0, "N*m"),
    "J*s": (1.0, "J*s"),
    "N*m^2/C^2": (1.0, "N*m^2/C^2"),
    "ohm*m": (1.0, "ohm*m"),
    "ohm*mm^2/m": (1e-6, "ohm*m"),
    "ohm*cm^2/m": (1e-4, "ohm*m"),
    "kg*m/s^2": (1.0, "N"),
}


def normalize_unit(unit: Any) -> Tuple[float, str, str]:
    """Return ``(factor_to_SI, si_unit, status)`` for a unit string."""
    cleaned = canonicalize_unit_text(unit)
    if cleaned in UNIT_TABLE:
        factor, si_unit = UNIT_TABLE[cleaned]
        return factor, si_unit, "ok"
    return 1.0, cleaned, "unknown_unit"


def normalize_quantity(value: Any, unit: Any) -> Dict[str, Any]:
    """Convert a numeric value and unit to SI-like canonical form.

    If the value is not numeric, it is kept unchanged and marked symbolic.
    """
    original_value = value
    original_unit = "" if unit is None else str(unit)
    numeric_value = to_float(value)
    factor, si_unit, unit_status = normalize_unit(unit)

    if numeric_value is not None:
        return {
            "value": float(f"{numeric_value * factor:.15g}"),
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


def value_unit_to_si(value: Any, unit: Any) -> Tuple[Optional[float], str, str]:
    norm = normalize_quantity(value, unit)
    if not norm["is_numeric"]:
        return None, norm["unit"], norm["status"]
    return norm["value"], norm["unit"], norm["status"]


def is_correct(
    pred: Optional[Dict[str, Any]],
    gold: Optional[Dict[str, Any]],
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-9,
) -> Tuple[bool, str]:
    if gold is None:
        return False, "missing_gold"
    if pred is None:
        return False, "no_prediction"

    pred_value, pred_unit, pred_status = value_unit_to_si(pred.get("answer"), pred.get("unit"))
    gold_value, gold_unit, gold_status = value_unit_to_si(gold.get("answer"), gold.get("unit"))

    if pred_value is None:
        return False, "answer_not_numeric"
    if gold_value is None:
        return False, "gold_answer_not_numeric"

    pred_unit_cmp = canonicalize_unit_text(pred_unit, collapse_electric_field=True)
    gold_unit_cmp = canonicalize_unit_text(gold_unit, collapse_electric_field=True)
    value_ok = math.isclose(pred_value, gold_value, rel_tol=rel_tol, abs_tol=abs_tol)
    unit_ok = pred_unit_cmp == gold_unit_cmp

    if value_ok and unit_ok:
        return True, "ok"

    raw_expected = f"{gold.get('answer')} {gold.get('unit')}".strip()
    raw_got = f"{pred.get('answer')} {pred.get('unit')}".strip()
    expected_si = f"{gold_value} {gold_unit}".strip()
    got_si = f"{pred_value} {pred_unit}".strip()

    if not value_ok and not unit_ok:
        return False, f"wrong_value_and_unit expected={expected_si}, got={got_si}; raw_expected={raw_expected}, raw_got={raw_got}"
    if not value_ok:
        return False, f"wrong_value expected={expected_si}, got={got_si}; raw_expected={raw_expected}, raw_got={raw_got}"
    return False, f"wrong_unit expected={gold_unit}, got={pred_unit}; raw_expected={raw_expected}, raw_got={raw_got}"


def run_one_sample(client: QwenSymCodeClient, sample: Dict[str, Any], max_code_retries: int, exec_timeout: int, gold_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    qid = qid_of(sample)
    raw = client.generate(build_symcode_messages(sample))
    code_text = extract_python_code(raw)
    execution = execute_symcode(code_text, timeout=exec_timeout)
    attempts = [{"kind": "initial", "raw_model_output": raw, "symcode": code_text, "execution": execution}]

    retry_idx = 0
    while execution["status"] != "pass" and retry_idx < max_code_retries:
        retry_idx += 1
        raw_debug = client.generate(build_debug_messages(sample, code_text, execution))
        code_text = extract_python_code(raw_debug)
        execution = execute_symcode(code_text, timeout=exec_timeout)
        attempts.append({"kind": f"debug_{retry_idx}", "raw_model_output": raw_debug, "symcode": code_text, "execution": execution})

    pred = execution.get("prediction")
    gold = gold_map.get(qid)
    correct, reason = is_correct(pred, gold)
    return {
        "query_id": qid,
        "query": query_of(sample),
        "symcode": code_text,
        "execution": execution,
        "prediction": pred,
        "answer": None if pred is None else pred.get("answer"),
        "unit": None if pred is None else pred.get("unit"),
        "gold": gold,
        "gold_answer": None if gold is None else gold.get("answer"),
        "gold_unit": None if gold is None else gold.get("unit"),
        "correct": correct,
        "reason": reason,
        "attempts": attempts,
        "num_attempts": len(attempts),
        "debugged": len(attempts) > 1,
    }


def save_outputs(results: List[Dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> None:
    num_correct = sum(1 for r in results if r["correct"])
    accuracy = num_correct / len(results) if results else 0.0

    result_log = {
        "model": args.model,
        "load_in_4bit": args.load_in_4bit,
        "pipeline": "question_to_symcode_to_execution_to_json",
        "num_samples": len(results),
        "num_correct": num_correct,
        "accuracy": accuracy,
        "results": results,
    }

    result_json = output_dir / args.output_json
    submission_json = output_dir / args.submission_json

    result_json.write_text(
        json.dumps(result_log, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    submission = [
        {
            "query_id": r["query_id"],
            "answer": r["answer"],
            "unit": r["unit"],
        }
        for r in results
    ]

    submission_json.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nDone.")
    print(f"Correct: {num_correct}/{len(results)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved result log JSON: {result_json}")
    print(f"Saved submission JSON: {submission_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--input", required=True, help="Input data file: JSONL, JSON object, or JSON array. No built-in 25 samples in this version.")
    parser.add_argument("--output-dir", default="/kaggle/working" if Path("/kaggle/working").exists() else ".")
    parser.add_argument("--output-json", default="result.json")
    parser.add_argument("--submission-json", default="submission.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-code-retries", type=int, default=1)
    parser.add_argument("--exec-timeout", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit BitsAndBytes quantization.")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.input)
    gold_map = build_gold_map(records)
    print(f"Records: {len(records)}")
    print(f"Gold labels: {len(gold_map)}")
    print(f"Model: {args.model}")
    print(f"4-bit: {args.load_in_4bit}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    client = QwenSymCodeClient(
        model_name=args.model,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    results: List[Dict[str, Any]] = []
    for idx, sample in enumerate(records, start=1):
        qid = qid_of(sample)
        print(f"[{idx:02d}/{len(records)}] {qid}")
        row = run_one_sample(client, sample, args.max_code_retries, args.exec_timeout, gold_map)
        results.append(row)
        exe = row["execution"]
        print("  prediction:", row["prediction"])
        print("  status:", exe["status"], "| stage:", exe["stage"], "| error:", exe["error"])
        print("  correct:", row["correct"], "|", row["reason"])

    save_outputs(results, args, output_dir)


if __name__ == "__main__":
    main()
