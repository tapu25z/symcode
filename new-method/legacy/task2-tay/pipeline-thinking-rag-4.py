#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Qwen3-8B Physics SymCode Pipeline + EXACT Type 2 Output

Pipeline:
  JSON question -> retrieve solver cards -> optional planner -> no-think codegen -> safety check -> execute code -> parse answer/unit -> no-think explanation -> EXACT output schema

Kaggle install:
  pip install -U "transformers>=4.51.0" accelerate bitsandbytes sympy torch
  # optional for embedding RAG:
  pip install -U sentence-transformers

Kaggle run:
  python symcode_thinking_hybrid_rag.py --input /kaggle/input/.../test04_easy.jsonl --rag-file rag_physics_kb_final_geometry_plus.jsonl --load-in-4bit --enable-thinking

Outputs:
  result.json
  submission.json
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def disable_broken_torchvision() -> None:
    """Prevent text-only Transformers jobs from importing broken torchvision.

    Some Kaggle images ship with a torchvision build that is incompatible with the
    installed torch build. Qwen3 is a text-only model, but recent Transformers
    may still check vision utilities while resolving model classes. If torchvision
    is broken, that optional import can crash AutoModelForCausalLM. This guard
    disables torchvision inside Transformers without affecting torch/CUDA.
    """
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("DISABLE_TRANSFORMERS_TORCHVISION", "1")
    try:
        import transformers.utils.import_utils as _tf_import_utils  # type: ignore
        if hasattr(_tf_import_utils, "_torchvision_available"):
            _tf_import_utils._torchvision_available = False
        if hasattr(_tf_import_utils, "_torchvision_version"):
            _tf_import_utils._torchvision_version = "N/A"
    except Exception:
        pass


disable_broken_torchvision()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

disable_broken_torchvision()


# No built-in TEST_DATA/GOLD. This version reads samples and gold labels from --input.

SYMCODE_FORMAT_EXAMPLES = r"""
SymCode format examples.
Use these examples only to learn the required output format and robust Python/SymPy style.
Do not copy numbers from examples unless the input problem has exactly the same numbers.
Detailed physics formulas, helper functions, and geometry construction patterns should come from retrieved RAG cards.

Example 1: scalar numeric compute
```python
import json
import sympy as sp

C = sp.Float(47e-6)
U = sp.Float(12)

E = sp.Rational(1, 2) * C * U**2

final_answer = float(E)
final_unit = "J"
print(json.dumps({"answer": final_answer, "unit": final_unit}))
```

Example 2: vector magnitude compute
```python
import json
import sympy as sp

Fx = sp.Float(3)
Fy = sp.Float(4)

F = sp.sqrt(Fx**2 + Fy**2)

final_answer = float(F)
final_unit = "N"
print(json.dumps({"answer": final_answer, "unit": final_unit}))
```

Example 3: symbolic solve then numeric answer
```python
import json
import sympy as sp

R = sp.Float(30)
X = sp.symbols("X", positive=True)

eq = sp.Eq(sp.sqrt(R**2 + X**2), 2 * R)
sol = sp.solve(eq, X)
X_value = [s for s in sol if s.is_real and s > 0][0]

final_answer = float(X_value)
final_unit = "ohm"
print(json.dumps({"answer": final_answer, "unit": final_unit}))
```
"""

PLANNER_SYSTEM_PROMPT = r"""You are a careful physics planner.

Think through the problem, then output a compact plan that will be used by a separate code generator.
The plan must be short and practical. Do not generate Python code.

After any internal thinking, output ONLY this JSON object:
{
  "case": "short physics case name",
  "knowns_si": ["key converted quantities"],
  "formula": "main formula or method",
  "steps": ["step 1", "step 2"],
  "target_unit": "ASCII unit",
  "pitfalls": ["unit/vector/sign/rounding pitfalls"]
}

Keep the final JSON under 120 words.
"""

SYMCODE_SYSTEM_PROMPT = r"""You are an expert physics solver and deterministic Python/SymPy code generator.

Given one physics problem JSON, optional retrieved solver cards, and optional planner notes, return ONLY executable Python code.
Do not write explanations. Do not output <think> tags.
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
- Use the planner notes if they are available, but verify them against the problem.
- Use retrieved solver cards only when they match the problem; ignore irrelevant cards.
- If a geometry helper card is provided, use it to construct coordinates/components before applying physics formulas.
- First decide the physics case/formula, then generate short deterministic code.
- The printed JSON must have exactly keys "answer" and "unit".
- final_answer must be a Python int or float, not a SymPy object.
- final_unit must be ASCII only.
- Use SI units unless the problem clearly asks for another target unit.
- For lens/image distance problems using cm in the problem, output image distance in cm.
- Convert prefixes correctly: u = micro = 1e-6, m = milli = 1e-3, n = nano = 1e-9, cm = 1e-2, mm = 1e-3.
- Use school-style electrostatics constant k = 9e9 unless the problem explicitly gives another value.
- Use epsilon_0 = 8.85e-12 only for capacitor/parallel-plate/permittivity formulas.
- Never call sp.Float with two positional arguments.
- Wrong: sp.Float(8.85, 12), sp.Float(100, -2).
- Correct: sp.Float(8.85e-12), sp.Float(100e-4), sp.Float(2e-8).
- For force/electric-field vectors, add components, not magnitudes. Equal opposite vectors cancel to zero.
- Use "ohm" instead of Ω.
- Use "m/s^2" instead of m/s².
- Use "N/C" or "V/m" for electric field.
- Do not use input(), open(), eval(), exec(), os, sys, subprocess, requests, socket, pathlib, shutil.
- Do not read or write files.
- Keep the code short and deterministic.

""" + SYMCODE_FORMAT_EXAMPLES

DEBUG_SYSTEM_PROMPT = r"""You are fixing Python/SymPy code for a physics problem.

Return ONLY corrected executable Python code.
Do not write explanations.
Do not output <think> tags.
Do not include markdown unless it is a Python code fence.

The corrected code MUST:
1. import json
2. import sympy as sp
3. set final_answer as a Python int or float
4. set final_unit as an ASCII string
5. print exactly one JSON object:
   print(json.dumps({"answer": final_answer, "unit": final_unit}))

Fix the shown failure only:
- syntax error
- safety error
- runtime error
- parse error
- wrong JSON keys
- non-numeric answer
- non-ASCII unit

Do not use input(), open(), eval(), exec(), os, sys, subprocess, requests, socket, pathlib, shutil.
Do not read or write files.
"""

EXPLANATION_SYSTEM_PROMPT = r"""You generate short explanations for solved Type 2 physics problems.

Return ONLY exactly 3 lines:
#step1: ...
#step2: ...
#step3: ...

Rules:
- Do not change the final answer or unit.
- Do not output JSON.
- Do not output markdown.
- Do not output code fences.
- Do not output <think> tags.
- Keep each step short and factual.
- #step1 should identify the physics method or formula.
- #step2 should describe substitution or computation.
- #step3 should state the final result using the given answer and unit.
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
    # Common LaTeX notation mapping fallback. The official mapping CSV may
    # already replace these, but keeping this here makes local tests robust.
    q = q.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    q = q.replace("\\leq", "<=").replace("\\geq", ">=").replace("\\approx", "approx")
    q = q.replace("\\Omega", "ohm").replace("\\mu", "u")
    q = q.replace("\\degree", "deg").replace("\\%", "%")
    q = re.sub(r"(?<=\d)\s*percent\b", "%", q, flags=re.IGNORECASE)
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
        "query_id", "id",
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




# -----------------------------
# Hybrid RAG: physics cards + gated geometry cards
# -----------------------------
# The KB is intentionally external. Use --rag-file rag_physics_kb_final_geometry_plus.jsonl.
# RAG card schema supports:
#   id, type=(policy|physics|geometry), domain, geometry_type, requires_geometry,
#   title, keywords, text.
# Optional embedding retrieval is used only if sentence-transformers is installed;
# otherwise the retriever falls back to lexical+rule scoring.


def _rag_norm_text(text: Any) -> str:
    return normalize_question_text(text).lower()


def _rag_tokens(text: Any) -> List[str]:
    text = _rag_norm_text(text)
    return re.findall(r"[a-zA-Z0-9_./^*+-]+", text)


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y"}
    return bool(x)


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    return [str(x)] if str(x).strip() else []


def load_rag_cards(rag_file: Optional[str]) -> List[Dict[str, Any]]:
    """Load the external RAG KB.

    Supported formats:
      - .jsonl: one card per line
      - .json: list of cards or one card
      - .txt/.md: blocks split by blank lines; each block becomes one card

    This function does NOT add any built-in cards. Passing the final KB file keeps
    the pipeline clean and makes RAG easy to audit.
    """
    if not rag_file:
        return []

    path = Path(rag_file)
    if not path.exists():
        # convenience: try file next to this script
        alt = Path(__file__).resolve().parent / rag_file
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"RAG file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    raw_cards: List[Dict[str, Any]] = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"RAG JSONL line {line_no} must be an object")
            raw_cards.append(obj)
    elif suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            raw_cards = [data]
        elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
            raw_cards = data
        else:
            raise ValueError("RAG JSON must be an object or list of objects")
    else:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        for i, block in enumerate(blocks, start=1):
            first_line = block.splitlines()[0].strip()
            raw_cards.append({
                "id": f"txt_card_{i}",
                "type": "physics",
                "domain": "general",
                "title": first_line[:80],
                "keywords": first_line,
                "text": block,
            })

    cards: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for i, card in enumerate(raw_cards, start=1):
        card_text = str(card.get("text") or card.get("content") or card.get("body") or "").strip()
        if not card_text:
            continue
        cid = str(card.get("id") or f"card_{i}").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out = dict(card)
        out["id"] = cid
        out["title"] = str(card.get("title") or card.get("name") or cid).strip()
        out["keywords"] = str(card.get("keywords") or card.get("tags") or "").strip()
        out["text"] = card_text
        out["type"] = str(card.get("type") or "physics").strip().lower()
        out["domain"] = str(card.get("domain") or "general").strip().lower()
        out["requires_geometry"] = _as_bool(card.get("requires_geometry", False))
        out["geometry_type"] = _as_list(card.get("geometry_type"))
        cards.append(out)
    return cards


class HybridRAGRetriever:
    def __init__(
        self,
        cards: List[Dict[str, Any]],
        mode: str = "hybrid",
        embedding_model: str = "intfloat/e5-small-v2",
        physics_top_k: int = 2,
        geometry_top_k: int = 1,
        max_cards: int = 4,
        include_unit_card: bool = True,
    ) -> None:
        self.cards = cards
        self.mode = mode
        self.embedding_model_name = embedding_model
        self.physics_top_k = max(0, physics_top_k)
        self.geometry_top_k = max(0, geometry_top_k)
        self.max_cards = max(0, max_cards)
        self.include_unit_card = include_unit_card
        self.embedder = None
        self.card_embeddings = None
        self.embedding_error = ""

        self._card_texts = [self._card_search_text(c) for c in self.cards]
        if self.mode in {"embedding", "hybrid"} and self.cards:
            self._try_build_embeddings()

    def _try_build_embeddings(self) -> None:
        try:
            disable_broken_torchvision()
            from sentence_transformers import SentenceTransformer  # type: ignore
            disable_broken_torchvision()
            self.embedder = SentenceTransformer(self.embedding_model_name)
            passages = ["passage: " + t for t in self._card_texts]
            self.card_embeddings = self.embedder.encode(
                passages,
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # optional dependency / offline model cache
            self.embedder = None
            self.card_embeddings = None
            self.embedding_error = f"embedding_unavailable:{type(exc).__name__}: {exc}"
            if self.mode == "embedding":
                # hard fallback rather than crashing long Kaggle jobs
                self.mode = "lexical"
            elif self.mode == "hybrid":
                self.mode = "lexical"

    @staticmethod
    def _card_search_text(card: Dict[str, Any]) -> str:
        return " ".join([
            str(card.get("title", "")),
            str(card.get("keywords", "")),
            str(card.get("text", "")),
            str(card.get("domain", "")),
            " ".join(_as_list(card.get("geometry_type"))),
        ])

    def _query_text(self, sample: Dict[str, Any]) -> str:
        # Do not include query_id/id. RAG should match problem wording only.
        q = query_of(sample)
        nq = normalize_question_text(q)
        return " ".join([q, nq, str(sample.get("type", ""))])

    def detect_geometry_types(self, query_text: str) -> List[str]:
        q = _rag_norm_text(query_text)
        toks = set(_rag_tokens(q))
        found: List[str] = []

        def add(name: str) -> None:
            if name not in found:
                found.append(name)

        if any(p in q for p in ["same line", "straight line", "collinear", "on a line", "lies on", "between", "outside the segment"]):
            add("collinear")
        if any(p in q for p in ["midpoint", "middle point", "perpendicular bisector", "equidistant", "same distance"]):
            add("midpoint_bisector")
        if "equilateral" in q or "60 deg" in q or "60deg" in q:
            add("equilateral")
        if "square" in q or "rectangle" in q or "four corners" in q or "vertices" in q or "diagonal" in q:
            add("square_rectangle")
        if "ring" in q or "circle" in q or ("axis" in q and "radius" in q) or "center of" in q:
            add("circle_axis")
        if "mirror" in q and any(p in q for p in ["tilt", "tilted", "rotated", "reflection", "reflected"]):
            add("mirror_tilt")
        if any(p in q for p in ["right triangle", "perpendicular", "90 deg", "90deg", "horizontal", "vertical", "projection"]):
            add("right_triangle")
        if any(p in q for p in ["angle", "inclined", "makes an angle", "components", "resultant", "law of cosines"]):
            add("angle_components")

        # AB/AC/BC three-side geometry. Require at least two side labels plus distance wording.
        side_hits = len({"ab", "ac", "bc", "ca", "cb"} & toks)
        if "triangle" in q or (side_hits >= 2 and any(w in q for w in ["distance", "apart", "from", "side"])):
            add("triangle_three_sides")

        # Fallback for any multi-point vector geometry.
        if len({"a", "b", "c"} & toks) >= 2 and any(w in q for w in ["point", "placed", "distance", "force", "field", "vector"]):
            add("coordinate_vector")

        return found

    def detect_domains(self, query_text: str) -> List[str]:
        q = _rag_norm_text(query_text)
        domains: List[str] = []

        def add(d: str) -> None:
            if d not in domains:
                domains.append(d)

        if any(w in q for w in ["charge", "electric", "capacitor", "capacitance", "voltage", "current", "resistor", "ohm", "rlc", "inductor", "magnetic", "solenoid", "transformer", "emf"]):
            add("electricity")
        if any(w in q for w in ["velocity", "speed", "acceleration", "force", "mass", "motion", "boat", "elevator", "spring", "height", "drop", "kinetic", "work"]):
            add("mechanics")
        if any(w in q for w in ["sound", "wave", "frequency", "wavelength", "harmonic"]):
            add("waves")
        if any(w in q for w in ["heat", "temperature", "gas", "thermal", "pressure", "density"]):
            add("thermal")
        if any(w in q for w in ["lens", "mirror", "focal", "image", "object distance"]):
            add("optics")
        if any(w in q for w in ["error", "uncertainty", "measurement", "least count"]):
            add("measurement")
        return domains

    def lexical_score(self, query_text: str, card: Dict[str, Any]) -> float:
        q_tokens = _rag_tokens(query_text)
        if not q_tokens:
            return 0.0
        q_counts: Dict[str, int] = {}
        for tok in q_tokens:
            if len(tok) <= 1:
                continue
            q_counts[tok] = q_counts.get(tok, 0) + 1

        title_tokens = set(_rag_tokens(card.get("title", "")))
        kw_tokens = set(_rag_tokens(card.get("keywords", "")))
        text_tokens = set(_rag_tokens(card.get("text", "")))
        raw = 0.0
        for tok, cnt in q_counts.items():
            if tok in kw_tokens:
                raw += 4.0 * cnt
            elif tok in title_tokens:
                raw += 2.5 * cnt
            elif tok in text_tokens:
                raw += 1.0 * cnt

        q_lower = _rag_norm_text(query_text)
        hay = _rag_norm_text(self._card_search_text(card))
        phrases = [
            "electric field", "net electric force", "point charges", "parallel plate",
            "capacitors in series", "connected in series", "connected in parallel",
            "resonance", "at resonance", "current is halved", "inductive reactance",
            "image distance", "focal length", "upstream", "downstream",
            "midpoint", "perpendicular bisector", "equilateral", "square", "rectangle",
        ]
        for phrase in phrases:
            if phrase in q_lower and phrase in hay:
                raw += 8.0
        # squash to 0..1-ish
        return raw / (raw + 25.0) if raw > 0 else 0.0

    def rule_score(self, query_text: str, card: Dict[str, Any], geometry_types: List[str], domains: List[str]) -> float:
        q = _rag_norm_text(query_text)
        cid = str(card.get("id", ""))
        ctype = str(card.get("type", "physics"))
        score = 0.0

        if ctype == "policy":
            return 1.0 if "unit" in cid else 0.4

        if ctype == "geometry":
            c_gtypes = _as_list(card.get("geometry_type"))
            if not geometry_types:
                return -1.0
            if any(gt in geometry_types for gt in c_gtypes):
                score += 1.0
            elif "coordinate_vector" in c_gtypes and geometry_types:
                score += 0.65
            else:
                return -0.5
            return score

        # physics card domain match
        domain = str(card.get("domain", "general"))
        if domain in domains:
            score += 0.55
        if domain == "general":
            score += 0.10

        # Disambiguation bonuses among close electric cards.
        if "electric field" in q and "electric_field" in cid:
            score += 0.8
        if "force" in q and ("coulomb_force" in cid or "vector_resultant" in cid):
            score += 0.7
        if "potential" in q and "potential" in cid:
            score += 0.8
        if any(w in q for w in ["capacitor", "capacitance", "plate"] ) and any(w in cid for w in ["capacitor", "parallel_plate"]):
            score += 0.8
        if "resonance" in q and "rlc" in cid:
            score += 0.8
        if any(w in q for w in ["upstream", "downstream", "boat", "current"] ) and "boat" in cid:
            score += 0.8
        if any(w in q for w in ["lens", "image distance", "focal"] ) and "lens" in cid:
            score += 0.8
        if _as_bool(card.get("requires_geometry", False)) and geometry_types:
            c_gtypes = _as_list(card.get("geometry_type"))
            if not c_gtypes or any(gt in geometry_types for gt in c_gtypes) or "coordinate_vector" in geometry_types:
                score += 0.3
        return min(score, 1.5)

    def embedding_scores(self, query_text: str) -> Dict[int, float]:
        if self.embedder is None or self.card_embeddings is None:
            return {}
        try:
            q_emb = self.embedder.encode(
                ["query: " + query_text],
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            sims = (q_emb @ self.card_embeddings.T).squeeze(0)
            return {i: float(sims[i].item()) for i in range(len(self.cards))}
        except Exception:
            return {}

    def score_all(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        query_text = self._query_text(sample)
        geometry_types = self.detect_geometry_types(query_text)
        domains = self.detect_domains(query_text)
        emb_scores = self.embedding_scores(query_text) if self.mode in {"embedding", "hybrid"} else {}

        rows: List[Dict[str, Any]] = []
        for i, card in enumerate(self.cards):
            lex = self.lexical_score(query_text, card)
            rule = self.rule_score(query_text, card, geometry_types, domains)
            emb = emb_scores.get(i, 0.0)
            emb01 = max(0.0, min(1.0, (emb + 1.0) / 2.0)) if emb_scores else 0.0

            if self.mode == "embedding" and emb_scores:
                final = 0.75 * emb01 + 0.25 * max(rule, 0.0)
            elif self.mode == "hybrid" and emb_scores:
                final = 0.55 * emb01 + 0.30 * lex + 0.15 * max(rule, 0.0)
            else:
                final = 0.70 * lex + 0.30 * max(rule, 0.0)

            # hard gate: geometry cards must match geometry signal
            if str(card.get("type")) == "geometry" and rule < 0:
                final = -1.0

            rows.append({
                "card": card,
                "score": final,
                "lexical": lex,
                "embedding": emb,
                "rule": rule,
                "geometry_types": geometry_types,
                "domains": domains,
            })
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows

    def retrieve(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.max_cards <= 0 or not self.cards:
            return []
        rows = self.score_all(sample)
        if not rows:
            return []

        selected: List[Dict[str, Any]] = []
        used: set[str] = set()

        def add_row(row: Dict[str, Any]) -> None:
            card = dict(row["card"])
            cid = str(card.get("id"))
            if cid in used:
                return
            card["_rag_score"] = round(float(row["score"]), 4)
            card["_rag_lexical"] = round(float(row["lexical"]), 4)
            card["_rag_embedding"] = round(float(row["embedding"]), 4)
            card["_rag_rule"] = round(float(row["rule"]), 4)
            card["_detected_geometry_types"] = row["geometry_types"]
            card["_detected_domains"] = row["domains"]
            selected.append(card)
            used.add(cid)

        # Always include the unit policy if present. It is cheap and prevents many unit mistakes.
        if self.include_unit_card:
            for row in rows:
                card = row["card"]
                if card.get("type") == "policy" and "unit" in str(card.get("id", "")):
                    add_row(row)
                    break

        physics_rows = [r for r in rows if r["card"].get("type") not in {"policy", "geometry"} and r["score"] > 0.02]
        for row in physics_rows[: self.physics_top_k]:
            add_row(row)

        # Geometry is retrieved only when the query or selected physics cards indicate geometry.
        detected_geometry = rows[0].get("geometry_types", []) if rows else []
        selected_requires_geometry = any(_as_bool(c.get("requires_geometry")) for c in selected)
        need_geometry = bool(detected_geometry) or selected_requires_geometry
        if need_geometry and self.geometry_top_k > 0:
            geom_rows = [r for r in rows if r["card"].get("type") == "geometry" and r["score"] > 0.02]
            for row in geom_rows[: self.geometry_top_k]:
                add_row(row)

        return selected[: self.max_cards]

    def status(self) -> Dict[str, Any]:
        return {
            "num_cards": len(self.cards),
            "mode": self.mode,
            "embedding_model": self.embedding_model_name,
            "embedding_enabled": self.embedder is not None,
            "embedding_error": self.embedding_error,
            "physics_top_k": self.physics_top_k,
            "geometry_top_k": self.geometry_top_k,
            "max_cards": self.max_cards,
        }


def format_rag_context(cards: List[Dict[str, Any]]) -> str:
    if not cards:
        return "No solver cards retrieved. Solve from first principles."
    chunks: List[str] = []
    for idx, card in enumerate(cards, start=1):
        title = str(card.get("title", "")).strip()
        card_id = str(card.get("id", "")).strip()
        ctype = str(card.get("type", "")).strip()
        domain = str(card.get("domain", "")).strip()
        score = card.get("_rag_score", "")
        text = str(card.get("text", "")).strip()
        chunks.append(f"[{idx}] {title} ({card_id}; type={ctype}; domain={domain}; score={score})\n{text}")
    return "\n\n".join(chunks)


def build_ragged_payload(sample: Dict[str, Any], rag_cards: List[Dict[str, Any]]) -> str:
    payload_obj = make_model_sample(sample)
    payload = json.dumps(payload_obj, ensure_ascii=False, indent=2)
    rag_context = format_rag_context(rag_cards)
    return f"Retrieved solver cards:\n{rag_context}\n\nInput JSON:\n{payload}"


def build_planner_messages(sample: Dict[str, Any], rag_cards: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
    payload = build_ragged_payload(sample, rag_cards or [])
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": payload + "\n\nThink if needed, then return the compact JSON plan only. Do not write code."},
    ]


def clean_planner_note(raw_plan: str) -> str:
    """Return the visible planner JSON/notes after any Qwen <think> block.

If the model runs out of tokens inside an unclosed <think> block, there is no
reliable plan to pass to codegen, so return an empty note and let codegen solve
from the problem + RAG cards.
    """
    if not raw_plan or not raw_plan.strip():
        return ""
    text = raw_plan.strip()
    if "<think>" in text and "</think>" not in text:
        return ""
    text = remove_thinking(text)
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    # Keep notes bounded so the codegen prompt stays compact.
    return text[:2000].strip()


def build_symcode_messages(
    sample: Dict[str, Any],
    rag_cards: Optional[List[Dict[str, Any]]] = None,
    planner_note: str = "",
) -> List[Dict[str, str]]:
    payload = build_ragged_payload(sample, rag_cards or [])
    plan_block = planner_note.strip() or "Planner note unavailable. Solve directly from the problem and retrieved cards."
    user_text = f"{payload}\n\nPlanner note:\n{plan_block}\n\nReturn executable Python code only. Do not output <think> tags."
    return [
        {"role": "system", "content": SYMCODE_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def build_debug_messages(sample: Dict[str, Any], bad_code: str, execution: Dict[str, Any], rag_cards: Optional[List[Dict[str, Any]]] = None, planner_note: str = "") -> List[Dict[str, str]]:
    payload = build_ragged_payload(sample, rag_cards or [])
    plan_block = planner_note.strip() or "Planner note unavailable. Solve directly from the problem and retrieved cards."
    user_text = f"""{payload}

Planner note:
{plan_block}

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
        enable_thinking: bool,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.enable_thinking = enable_thinking

        disable_broken_torchvision()
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

        disable_broken_torchvision()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: Optional[bool] = None,
        max_new_tokens_override: Optional[int] = None,
    ) -> str:
        use_thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=use_thinking,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens_override or self.max_new_tokens),
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
    text = str(text or "")
    if "<think>" in text and "</think>" not in text:
        # Unclosed thinking means the model probably hit the token limit before
        # producing final visible output. Do not treat reasoning as code.
        return text.split("<think>", 1)[0].strip()
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
            if (
                isinstance(fn, ast.Attribute)
                and fn.attr == "Float"
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "sp"
                and len(node.args) >= 2
            ):
                return "bad_sp_float_scientific_notation:use sp.Float(8.85e-12), not sp.Float(8.85, 12)"
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



def format_answer_number(x: Any) -> str:
    """Format Type 2 answer as a numeric-only string.

    Keeps the value compact for JSON output. The unit is handled separately.
    """
    v = to_float(x)
    if v is None:
        return "0"
    if abs(v) < 1e-15:
        v = 0.0
    return f"{v:.12g}"


def ascii_unit_text(unit: Any) -> str:
    """Return an ASCII unit string for EXACT Type 2 output."""
    u = canonicalize_unit_text(unit)
    try:
        u.encode("ascii")
        return u
    except UnicodeEncodeError:
        return u.encode("ascii", "ignore").decode("ascii")


def compact_rag_cards_for_explanation(cards: Optional[List[Dict[str, Any]]], max_cards: int = 2, max_chars: int = 450) -> List[Dict[str, str]]:
    compact: List[Dict[str, str]] = []
    for c in (cards or [])[:max_cards]:
        compact.append({
            "title": str(c.get("title", "")),
            "domain": str(c.get("domain", "")),
            "text": str(c.get("text", ""))[:max_chars],
        })
    return compact


def build_explanation_messages(
    sample: Dict[str, Any],
    rag_cards: Optional[List[Dict[str, Any]]],
    planner_note: str,
    symcode: str,
    pred: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build a small no-gold payload for explanation generation."""
    problem_obj = make_model_sample(sample)
    answer = format_answer_number(pred.get("answer"))
    unit = ascii_unit_text(pred.get("unit"))
    compact_cards = compact_rag_cards_for_explanation(rag_cards)

    user_text = f"""
Problem JSON:
{json.dumps(problem_obj, ensure_ascii=False, indent=2)}

Retrieved solver cards:
{json.dumps(compact_cards, ensure_ascii=False, indent=2)}

Planner note:
{planner_note.strip() or "N/A"}

Executed SymCode:
```python
{symcode}
```

Final prediction:
answer = {answer}
unit = {unit}

Generate the explanation only in this exact format:
#step1: ...
#step2: ...
#step3: ...
""".strip()

    return [
        {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def fallback_explanation(pred: Optional[Dict[str, Any]]) -> str:
    ans = format_answer_number(pred.get("answer") if pred else None)
    unit = ascii_unit_text(pred.get("unit") if pred else "")
    suffix = f" {unit}" if unit else ""
    return (
        "#step1: Identify the relevant physics formula from the problem.\n"
        "#step2: Substitute the given values and compute the requested quantity using SymCode.\n"
        f"#step3: The computed result is {ans}{suffix}."
    )


def clean_explanation(text: str, fallback: str) -> str:
    """Force explanation into exactly #step1/#step2/#step3 lines."""
    text = remove_thinking(text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
    lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^#step[123]\s*:", line, flags=re.IGNORECASE):
            # Normalize tag casing/spaces.
            m = re.match(r"^#step([123])\s*:\s*(.*)$", line, flags=re.IGNORECASE)
            if m:
                lines.append(f"#step{m.group(1)}: {m.group(2).strip()}")
    if len(lines) >= 3:
        return "\n".join(lines[:3])
    return fallback


def symcode_to_reasoning(symcode: str) -> Dict[str, Any]:
    """Represent executed SymCode as the optional reasoning object."""
    steps: List[str] = []
    for line in str(symcode or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        steps.append(line)
    return {"type": "symcode", "steps": steps}


def build_type2_final_output(sample: Dict[str, Any], pred: Optional[Dict[str, Any]], explanation: str, symcode: str) -> Dict[str, Any]:
    """Build the EXACT Type 2 response object."""
    pred = pred if isinstance(pred, dict) else {"answer": 0, "unit": ""}
    return {
        "query_id": qid_of(sample),
        "answer": format_answer_number(pred.get("answer")),
        "unit": ascii_unit_text(pred.get("unit")),
        "explanation": explanation.strip() or fallback_explanation(pred),
        "premises_used": [],
        "reasoning": symcode_to_reasoning(symcode),
    }


def validate_type2_final_output(obj: Dict[str, Any]) -> Tuple[bool, str]:
    required = {"query_id", "answer", "unit", "explanation", "premises_used", "reasoning"}
    missing = required - set(obj.keys())
    if missing:
        return False, f"missing:{sorted(missing)}"
    if to_float(obj.get("answer")) is None:
        return False, "answer_not_numeric"
    try:
        str(obj.get("unit", "")).encode("ascii")
    except UnicodeEncodeError:
        return False, "unit_not_ascii"
    if not str(obj.get("explanation", "")).strip():
        return False, "empty_explanation"
    if obj.get("premises_used") != []:
        return False, "type2_premises_used_must_be_empty"
    if not isinstance(obj.get("reasoning"), dict):
        return False, "reasoning_not_object"
    return True, "ok"



def run_one_sample(
    client: QwenSymCodeClient,
    sample: Dict[str, Any],
    max_code_retries: int,
    exec_timeout: int,
    gold_map: Dict[str, Dict[str, Any]],
    retriever: HybridRAGRetriever,
    explain_max_new_tokens: int = 256,
) -> Dict[str, Any]:
    qid = qid_of(sample)
    rag_cards = retriever.retrieve(sample)

    raw_plan = ""
    planner_note = ""
    attempts: List[Dict[str, Any]] = []
    if client.enable_thinking:
        raw_plan = client.generate(build_planner_messages(sample, rag_cards), enable_thinking=True)
        planner_note = clean_planner_note(raw_plan)
        attempts.append({
            "kind": "planner_think",
            "raw_model_output": raw_plan,
            "planner_note": planner_note,
            "execution": {"status": "skip", "stage": "planner", "error": "", "stdout": "", "stderr": "", "prediction": None},
        })

    raw = client.generate(build_symcode_messages(sample, rag_cards, planner_note), enable_thinking=False)
    code_text = extract_python_code(raw)
    execution = execute_symcode(code_text, timeout=exec_timeout)
    attempts.append({"kind": "codegen_nothink", "raw_model_output": raw, "symcode": code_text, "execution": execution})

    retry_idx = 0
    while execution["status"] != "pass" and retry_idx < max_code_retries:
        retry_idx += 1
        raw_debug = client.generate(build_debug_messages(sample, code_text, execution, rag_cards, planner_note), enable_thinking=False)
        code_text = extract_python_code(raw_debug)
        execution = execute_symcode(code_text, timeout=exec_timeout)
        attempts.append({"kind": f"debug_nothink_{retry_idx}", "raw_model_output": raw_debug, "symcode": code_text, "execution": execution})

    pred = execution.get("prediction")

    explanation = fallback_explanation(pred)
    raw_explanation = ""
    if pred is not None and execution.get("status") == "pass":
        try:
            raw_explanation = client.generate(
                build_explanation_messages(sample, rag_cards, planner_note, code_text, pred),
                enable_thinking=False,
                max_new_tokens_override=explain_max_new_tokens,
            )
            explanation = clean_explanation(raw_explanation, fallback=explanation)
        except Exception as exc:
            raw_explanation = f"explanation_generation_failed:{type(exc).__name__}: {exc}"
            explanation = fallback_explanation(pred)

    final_output = build_type2_final_output(sample, pred, explanation, code_text)
    final_ok, final_reason = validate_type2_final_output(final_output)

    gold = gold_map.get(qid)
    correct, reason = is_correct(pred, gold)
    return {
        "query_id": qid,
        "query": query_of(sample),
        "rag_cards": [
            {
                "id": c.get("id"),
                "title": c.get("title"),
                "type": c.get("type"),
                "domain": c.get("domain"),
                "score": c.get("_rag_score"),
                "rule": c.get("_rag_rule"),
                "lexical": c.get("_rag_lexical"),
                "embedding": c.get("_rag_embedding"),
            }
            for c in rag_cards
        ],
        "detected_geometry_types": rag_cards[0].get("_detected_geometry_types", []) if rag_cards else [],
        "detected_domains": rag_cards[0].get("_detected_domains", []) if rag_cards else [],
        "planner_note": planner_note,
        "symcode": code_text,
        "execution": execution,
        "prediction": pred,
        "answer": final_output["answer"],
        "unit": final_output["unit"],
        "explanation": explanation,
        "raw_explanation": raw_explanation,
        "premises_used": [],
        "reasoning": final_output["reasoning"],
        "final_output": final_output,
        "final_output_valid": final_ok,
        "final_output_reason": final_reason,
        "gold": gold,
        "gold_answer": None if gold is None else gold.get("answer"),
        "gold_unit": None if gold is None else gold.get("unit"),
        "correct": correct,
        "reason": reason,
        "attempts": attempts,
        "num_attempts": len(attempts),
        "debugged": any(str(a.get("kind", "")).startswith("debug") for a in attempts),
    }


def save_outputs(results: List[Dict[str, Any]], args: argparse.Namespace, output_dir: Path, retriever_status: Dict[str, Any]) -> None:
    num_correct = sum(1 for r in results if r["correct"])
    accuracy = num_correct / len(results) if results else 0.0

    result_log = {
        "model": args.model,
        "load_in_4bit": args.load_in_4bit,
        "planner_thinking": args.enable_thinking,
        "codegen_thinking": False,
        "rag_file": args.rag_file,
        "rag": retriever_status,
        "pipeline": "fast_exact_type2_symcode_explain",
        "num_samples": len(results),
        "num_correct": num_correct,
        "accuracy": accuracy,
        "results": results,
    }

    result_json = output_dir / args.output_json
    submission_json = output_dir / args.submission_json

    result_json.write_text(json.dumps(result_log, ensure_ascii=False, indent=2), encoding="utf-8")

    submission = [
        r.get("final_output") or {
            "query_id": r.get("query_id", ""),
            "answer": format_answer_number(r.get("answer")),
            "unit": ascii_unit_text(r.get("unit")),
            "explanation": r.get("explanation") or fallback_explanation(r.get("prediction")),
            "premises_used": [],
            "reasoning": r.get("reasoning") or symcode_to_reasoning(str(r.get("symcode", ""))),
        }
        for r in results
    ]
    submission_json.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Correct: {num_correct}/{len(results)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved result log JSON: {result_json}")
    print(f"Saved submission JSON: {submission_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--input", required=True, help="Input data file: JSONL, JSON object, or JSON array.")
    parser.add_argument("--output-dir", default="/kaggle/working" if Path("/kaggle/working").exists() else ".")
    parser.add_argument("--output-json", default="result.json")
    parser.add_argument("--submission-json", default="submission.json")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-code-retries", type=int, default=3)
    parser.add_argument("--exec-timeout", type=int, default=8)
    parser.add_argument("--explain-max-new-tokens", type=int, default=256, help="Max tokens for the no-thinking explanation generation call.")

    parser.add_argument("--rag-file", default="rag_physics_kb_final_geometry_plus.jsonl", help="Final RAG KB jsonl/json/txt/md. If relative, also searches next to this script.")
    parser.add_argument("--rag-mode", choices=["lexical", "embedding", "hybrid"], default="lexical", help="hybrid uses embedding if sentence-transformers is available, otherwise lexical fallback.")
    parser.add_argument("--embedding-model", default="intfloat/e5-small-v2")
    parser.add_argument("--physics-rag-top-k", type=int, default=2)
    parser.add_argument("--geometry-rag-top-k", type=int, default=1)
    parser.add_argument("--rag-max-cards", type=int, default=4)
    parser.add_argument("--no-unit-card", dest="include_unit_card", action="store_false")

    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", help="Enable Qwen3 thinking mode for the planner phase only. Codegen/debug always run with thinking disabled.")
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false", help="Skip the thinking planner phase and run direct no-thinking codegen.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit BitsAndBytes quantization.")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=True, enable_thinking=False, include_unit_card=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.input)
    gold_map = build_gold_map(records)
    rag_cards_db = load_rag_cards(args.rag_file)
    retriever = HybridRAGRetriever(
        rag_cards_db,
        mode=args.rag_mode,
        embedding_model=args.embedding_model,
        physics_top_k=args.physics_rag_top_k,
        geometry_top_k=args.geometry_rag_top_k,
        max_cards=args.rag_max_cards,
        include_unit_card=args.include_unit_card,
    )

    print(f"Records: {len(records)}")
    print(f"Gold labels: {len(gold_map)}")
    print(f"RAG: {retriever.status()}")
    print(f"Model: {args.model}")
    print(f"4-bit: {args.load_in_4bit}")
    print(f"Planner thinking: {args.enable_thinking}")
    print("Codegen thinking: False")
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
        enable_thinking=args.enable_thinking,
    )

    results: List[Dict[str, Any]] = []
    for idx, sample in enumerate(records, start=1):
        qid = qid_of(sample)
        print(f"[{idx:02d}/{len(records)}] {qid}")
        row = run_one_sample(client, sample, args.max_code_retries, args.exec_timeout, gold_map, retriever, args.explain_max_new_tokens)
        results.append(row)
        exe = row["execution"]
        print("  domains:", row.get("detected_domains"), "| geometry:", row.get("detected_geometry_types"))
        print("  rag:", row.get("rag_cards"))
        print("  prediction:", row["prediction"])
        print("  final_output_valid:", row.get("final_output_valid"), "|", row.get("final_output_reason"))
        print("  status:", exe["status"], "| stage:", exe["stage"], "| error:", exe["error"])
        print("  correct:", row["correct"], "|", row["reason"])

    save_outputs(results, args, output_dir, retriever.status())


if __name__ == "__main__":
    main()