#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module 2: chuẩn hóa câu hỏi và trích xuất given/conditions/target.

File này chứa các heuristic nhẹ trước inference, router answer_type, parser JSON
trả về từ LLM, và hàm generate_module2 gọi Ollama có retry.
"""

from __future__ import annotations

import json
import re
import time
import hashlib
from typing import Any, Dict, List, Optional

from .llm import chat_messages
from .prompts import MODULE_2_SYSTEM, PROBLEM_TYPE_SYSTEM
from .debug_trace import compact_examples, log_json


VALID_ANSWER_TYPES = {"numeric_compute"}
GENERIC_PROBLEM_TYPE_VALUES = {
    "type",
    "type1",
    "type2",
    "task",
    "task1",
    "task2",
    "public",
    "private",
    "train",
    "test",
    "unknown",
}


# ===== Extracted notebook cell 18 =====
# Cleans up question text by normalizing whitespace and
# replacing common Unicode characters with ASCII equivalents.

def normalize_question_text(q: Any) -> str:
    """
    Normalize a question string:
      - Replace non-breaking spaces, zero-width chars
      - Replace Unicode math notation with stable ASCII/LaTeX-like notation
      - Collapse whitespace
    """
    q = str(q)
    q = q.replace("\u00a0", " ")     # non-breaking space
    q = q.replace("\u200b", "")      # zero-width space
    q = q.replace("×", "*")          # multiplication sign
    q = q.replace("·", "*").replace("⋅", "*")
    q = q.replace("−", "-")          # minus sign
    q = q.replace("–", "-")          # en-dash
    q = q.replace("—", "-")          # em-dash
    q = q.replace("±", "+/-")
    q = q.replace("π", r"\pi")
    q = q.replace("Ω", "ohm").replace("Ω", "ohm")
    q = q.replace("μ", "u").replace("µ", "u")
    q = q.replace("°C", "degC")
    q = q.replace("°", "deg")
    q = q.translate(str.maketrans({
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
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }))
    q = re.sub(r"\s+", " ", q).strip()
    return q


# ===== Extracted notebook cell 20 =====
def infer_problem_type(record: Dict[str, Any]) -> str:
    """
    Return explicit problem type metadata when it exists.

    LLM classification is handled by generate_problem_type() during inference.
    This helper intentionally avoids keyword or ID-prefix heuristics so that
    few-shot routing is not silently tied to dataset-specific labels.
    """
    for key in ["problem_type", "type", "category", "topic", "formula_id", "label"]:
        val = record.get(key)
        if val:
            label = str(val).strip()
            label_norm = re.sub(r"[\s_-]+", "", label.lower())
            if label_norm in GENERIC_PROBLEM_TYPE_VALUES:
                continue
            if re.fullmatch(r"type\d+|task\d+", label_norm):
                continue
            if label.upper() == "QA":
                continue
            return label
    return "unknown"


def generate_problem_type(
    client: Any,
    model: str,
    question: str,
    candidate_examples: Any,
) -> str:
    """Classify the physics problem family with the LLM for few-shot routing."""
    if isinstance(candidate_examples, dict):
        base_candidates = sorted({str(x).strip() for x in candidate_examples.keys() if str(x).strip()})
        failed_types = ["LD", "DT", "DDT", "CHLT", "CH", "NL", "VT"]
        base_candidates = {label for label in base_candidates if label.upper() != "QA"}
        candidates = sorted(base_candidates, key=lambda x: (x not in failed_types, x))
        candidate_payload = []
        for label in candidates:
            examples = []
            for ex in candidate_examples.get(label, [])[:2]:
                if isinstance(ex, dict) and ex.get("question"):
                    examples.append(str(ex["question"])[:300])
            candidate_payload.append({"label": label, "examples": examples})
    else:
        candidates = sorted({
            str(x).strip()
            for x in candidate_examples
            if str(x).strip() and str(x).strip().upper() != "QA"
        })
        candidate_payload = [{"label": label, "examples": []} for label in candidates]

    if "unknown" not in candidates:
        candidates.append("unknown")
        candidate_payload.append({"label": "unknown", "examples": []})

    payload = {
        "candidates": candidate_payload,
        "question": question,
    }
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": PROBLEM_TYPE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]

    from kaggle_pipeline import core as runtime_config

    for attempt in range(1, runtime_config.MAX_RETRIES + 1):
        try:
            print(f"  Problem type: LLM attempt {attempt}/{runtime_config.MAX_RETRIES}", flush=True)
            content = chat_messages(client, model, messages)
            parsed = extract_json_block(content)
            problem_type = str((parsed or {}).get("problem_type", "")).strip()
            if problem_type in candidates:
                return problem_type
            print(f"  Problem type: invalid label on attempt {attempt}: {problem_type}", flush=True)
        except Exception as exc:
            print(f"  Problem type: error on attempt {attempt}: {exc}", flush=True)
        time.sleep(runtime_config.SLEEP_SECONDS)

    return "unknown"


def detect_answer_type(
    raw_question: str,
    module2_json: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Force all questions to numeric_compute as per the latest dataset trend.
    Type2 is always evaluated as a numeric computation task.
    """
    return "numeric_compute"


def force_numeric_answer_type(module2_json: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Type2 Module 2 targets so downstream routing is always numeric."""
    targets = module2_json.get("target", [])
    if isinstance(targets, list):
        for target in targets:
            if isinstance(target, dict):
                target["answer_type"] = "numeric_compute"
                target.pop("options", None)
    return module2_json


def has_explicit_mc_options(question: str) -> bool:
    """Nhận diện option trắc nghiệm thật, tránh để LLM tự bịa option."""
    q = str(question or "")

    # Dạng phổ biến: "A. ... B. ..." hoặc mỗi option ở đầu dòng.
    # Cố ý ưu tiên chữ hoa A-D để không nuốt "(a), (b)" trong đề nhiều ý.
    markers = re.findall(r"(?:^|[\n\r]|[;|])\s*([A-D])[\).:]\s+\S+", q)
    if len(set(markers)) >= 2:
        return True

    # Một số đề để option cùng dòng: "... A. foo B. bar C. baz".
    inline_markers = re.findall(r"(?<![A-Za-z0-9])([A-D])[\).:]\s+\S+", q)
    return len(set(inline_markers)) >= 3


def looks_numeric_value_question(question: str) -> bool:
    """Nhận diện câu hỏi value/magnitude của đại lượng vật lý dù đáp án là 0."""
    q = " ".join(str(question or "").lower().split())
    if has_explicit_mc_options(q):
        return False
    if re.search(r"\b(unit|si unit|s\.i\. unit|characteristic|category|kind|type)\b", q):
        return False

    value_patterns = [
        r"\bwhat\s+is\s+(the\s+)?(value|magnitude|amount)\s+of\b",
        r"\bfind\s+(the\s+)?(value|magnitude|amount)\s+of\b",
        r"\bdetermine\s+(the\s+)?(value|magnitude|amount)\s+of\b",
        r"\bcalculate\s+(the\s+)?(value|magnitude|amount)\s+of\b",
        r"\bhow\s+(much|many)\b",
    ]
    if not any(re.search(pattern, q) for pattern in value_patterns):
        return False

    physical_quantity_words = [
        "energy", "work", "power", "force", "current", "voltage", "resistance",
        "impedance", "reactance", "capacitance", "inductance", "charge",
        "field", "flux", "frequency", "period", "speed", "velocity",
        "acceleration", "distance", "length", "height", "mass", "time",
        "temperature", "pressure", "wavelength", "momentum", "torque",
    ]
    return any(word in q for word in physical_quantity_words)


# ===== Extracted notebook cell 22 =====
# Sends the question to the LLM with:
#   - System message: instructions + 5 static examples
#   - User/assistant turns: 2 dynamic few-shot examples
#   - Final user message: the actual question
# Expects JSON output with {given, conditions, target}.

_JSON_VALUE_FRACTION_RE = re.compile(
    r'("value"\s*:\s*)'
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*/\s*'
    r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)'
    r'(?=\s*[,}\]])'
)


def quote_bare_value_fractions(text: str) -> str:
    """
    Repair a common LLM JSON error: unquoted fractions in the value field.

    JSON does not allow `"value": 5/9`; downstream Module 3 can parse
    `"value": "5/9"`, so quote only this narrow value-field pattern.
    """
    def repl(match: re.Match[str]) -> str:
        fraction = re.sub(r"\s+", "", match.group(2))
        return f'{match.group(1)}"{fraction}"'

    return _JSON_VALUE_FRACTION_RE.sub(repl, text)


def loads_json_with_repairs(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        repaired = quote_bare_value_fractions(text)
        if repaired != text:
            return json.loads(repaired)
        raise


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from LLM output.
    Handles:
      - Fenced ```json ... ``` blocks
      - Raw JSON text
      - JSON embedded in other text (first { to last })
    """
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    # Try fenced block first
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    # Try direct parse
    try:
        return loads_json_with_repairs(text)
    except Exception:
        pass

    # Try extracting from first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return loads_json_with_repairs(text[start:end + 1])
        except Exception:
            return None

    return None


def module2_validation_error(obj: Any) -> str:
    """Return an empty string when Module 2 JSON matches the expected schema."""
    if not isinstance(obj, dict):
        return "root is not a JSON object"

    if set(obj.keys()) != {"given", "conditions", "target"}:
        missing = sorted({"given", "conditions", "target"} - set(obj.keys()))
        extra = sorted(set(obj.keys()) - {"given", "conditions", "target"})
        parts = []
        if missing:
            parts.append(f"missing top-level keys: {missing}")
        if extra:
            parts.append(f"extra top-level keys: {extra}")
        return "; ".join(parts)

    if not isinstance(obj["given"], list):
        return "given is not a list"
    if not isinstance(obj["conditions"], list):
        return "conditions is not a list"
    if not isinstance(obj["target"], list):
        return "target is not a list"

    # Validate each given item
    for idx, item in enumerate(obj["given"]):
        if not isinstance(item, dict):
            return f"given[{idx}] is not an object"
        for key in ["name", "symbol", "value", "unit"]:
            if key not in item:
                return f"given[{idx}] missing key: {key}"

    # Validate each target item
    for idx, item in enumerate(obj["target"]):
        if not isinstance(item, dict):
            return f"target[{idx}] is not an object"
        for key in ["name", "symbol", "unit", "answer_type"]:
            if key not in item:
                return f"target[{idx}] missing key: {key}"
        if item.get("answer_type") not in VALID_ANSWER_TYPES:
            return f"target[{idx}] invalid answer_type: {item.get('answer_type')!r}"
        if item.get("answer_type") == "multiple_choice":
            options = item.get("options")
            if not isinstance(options, dict) or len(options) < 2:
                return f"target[{idx}] multiple_choice missing valid options"

    return ""


def validate_module2(obj: Dict[str, Any]) -> bool:
    """
    Validate Module 2 output against the expected schema.

    Required structure:
      - Keys: exactly {given, conditions, target}
      - given: list of dicts, each with {name, symbol, value, unit}
      - conditions: list of strings
      - target: list of dicts, each with {name, symbol, unit, answer_type}
    """
    return module2_validation_error(obj) == ""


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _infer_answer_type(question: str, target: Dict[str, Any]) -> str:
    answer_type = str(target.get("answer_type", "")).strip()
    if answer_type in VALID_ANSWER_TYPES:
        return answer_type

    q = str(question or "").strip().lower()
    options = target.get("options")
    if isinstance(options, dict) and len(options) >= 2:
        return "multiple_choice"
    if re.search(r"\b(is|are|does|do|did|can|will|would|should)\b.*\?", q):
        return "yes_no"
    if str(target.get("unit", "")).strip():
        return "numeric_compute"
    if re.search(r"\b(calculate|find|determine|compute|how many|how much|what is the value|magnitude)\b", q):
        return "numeric_compute"
    return "conceptual_qa"


def repair_module2_schema(obj: Any, question: str) -> Optional[Dict[str, Any]]:
    """
    Normalize common JSON shape mistakes without solving the physics problem.

    This is intentionally schema-only:
      - unwrap nested module2_extract/output objects
      - ignore extra top-level keys
      - coerce dict target/given to singleton lists
      - fill missing string fields with ""
      - repair invalid/missing answer_type from question wording
    """
    if not isinstance(obj, dict):
        return None

    for nested_key in ("module2_extract", "output", "completion"):
        nested = obj.get(nested_key)
        if isinstance(nested, dict):
            obj = nested
            break

    given: List[Dict[str, Any]] = []
    for item in _as_list(obj.get("given")):
        if not isinstance(item, dict):
            continue
        given.append({
            "name": str(item.get("name", "")),
            "symbol": str(item.get("symbol", "")),
            "value": item.get("value", ""),
            "unit": str(item.get("unit", "")),
        })

    conditions = [
        str(item)
        for item in _as_list(obj.get("conditions"))
        if item is not None and str(item).strip()
    ]

    target: List[Dict[str, Any]] = []
    for item in _as_list(obj.get("target", obj.get("targets"))):
        if not isinstance(item, dict):
            continue
        fixed = {
            "name": str(item.get("name", "")),
            "symbol": str(item.get("symbol", "")),
            "unit": str(item.get("unit", "")),
            "answer_type": "",
        }
        fixed["answer_type"] = _infer_answer_type(question, {**item, **fixed})
        if fixed["answer_type"] == "multiple_choice" and isinstance(item.get("options"), dict):
            fixed["options"] = item["options"]
        target.append(fixed)

    repaired = {
        "given": given,
        "conditions": conditions,
        "target": target,
    }
    return repaired if validate_module2(repaired) else None


_RAW_QUANTITY_PATTERN = re.compile(
    r"""
    (?P<symbol>\b[A-Za-z][A-Za-z0-9_]*\b)
    \s*=\s*
    (?P<value>
        [+-]?
        (?:
            (?:\d+(?:\.\d*)?|\.\d+)
            (?:
                \s*(?:\*|x|×)\s*10\s*(?:\^|\*\*)\s*[+-]?\d+
                |[eE][+-]?\d+
            )?
        )
    )
    \s*
    (?P<unit>[A-Za-zµμΩ%°][A-Za-z0-9µμΩ%°/*.^·\-()²³]*)
    """,
    re.VERBOSE,
)


def extract_raw_quantity_hints(question: str) -> List[Dict[str, str]]:
    """Detect simple raw `symbol = value unit` quantities to guide Module 2."""
    hints: List[Dict[str, str]] = []
    seen = set()
    for match in _RAW_QUANTITY_PATTERN.finditer(str(question or "")):
        symbol = match.group("symbol").strip()
        raw_value = re.sub(r"\s+", " ", match.group("value").strip())
        raw_unit = match.group("unit").strip().rstrip(".,;:")
        raw_text = f"{symbol} = {raw_value} {raw_unit}"
        key = (symbol, raw_value, raw_unit)
        if key in seen:
            continue
        seen.add(key)
        hints.append({
            "symbol": symbol,
            "raw_value": raw_value,
            "raw_unit": raw_unit,
            "raw_text": raw_text,
        })
    return hints


def build_module2_question_prompt(question: str) -> str:
    """Append deterministic raw extraction hints when regex can find them."""
    hints = extract_raw_quantity_hints(question)
    if not hints:
        return question
    hint_payload = {
        "instruction": (
            "Use these regex-detected raw quantities as extraction hints. "
            "They are copied from the question before unit conversion."
        ),
        "raw_quantity_hints": hints,
    }
    return (
        f"{question}\n\n"
        "Regex-detected raw quantity hints:\n"
        f"{json.dumps(hint_payload, ensure_ascii=False, indent=2)}"
    )


def build_module2_few_shot_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Format a dynamic few-shot example for Module 2 log and prompt."""
    module2_ext = ex.get("module2_extract", {})
    return {
        "input": {
            "question": ex.get("question", "")
        },
        "output": {
            "given": module2_ext.get("given", []),
            "conditions": module2_ext.get("conditions", []),
            "target": module2_ext.get("target", [])
        }
    }


def generate_module2(
    client: Any,
    model: str,
    question: str,
    few_shot_examples: List[Dict[str, Any]],
    debug_hook: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Module 2: Extract {given, conditions, target} from a physics question.

    Uses few-shot prompting via chat messages:
      - system: instructions + 5 static examples (MODULE_2_SYSTEM)
      - user/assistant pairs: dynamic few-shot examples
      - user: the actual question

    Retries up to MAX_RETRIES times on failure.
    Returns None if all attempts fail.
    """
    # Build the message list
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": MODULE_2_SYSTEM},
    ]

    # Few-shot giúp giảm lỗi JSON rỗng/schema sai trên Kaggle.
    for ex in few_shot_examples:
        if not ex.get("question") or not ex.get("module2_extract"):
            continue
        ex_formatted = build_module2_few_shot_example(ex)
        messages.append({
            "role": "user",
            "content": json.dumps({"input": ex_formatted["input"]}, indent=2, ensure_ascii=False)
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps({"output": ex_formatted["output"]}, indent=2, ensure_ascii=False)
        })

    # Add the actual question as the final user message, wrapped in the input structure
    actual_query = json.dumps({
        "input": {
            "question": build_module2_question_prompt(question)
        }
    }, indent=2, ensure_ascii=False)
    messages.append({"role": "user", "content": actual_query})
    trace_id = _trace_id_from_question(question)
    log_json(f"{trace_id}_module2_prompt_messages", {
        "question": question,
        "fewshot_ids": [ex.get("id") for ex in few_shot_examples],
        "fewshots": [build_module2_few_shot_example(ex) for ex in few_shot_examples],
        "messages": messages,
    })

    from kaggle_pipeline import core as runtime_config

    # Retry loop
    for attempt in range(1, runtime_config.MAX_RETRIES + 1):
        try:
            print(f"  Module 2: LLM attempt {attempt}/{runtime_config.MAX_RETRIES}", flush=True)
            content = chat_messages(client, model, messages, use_adapter="module2")
            log_json(f"{trace_id}_module2_raw_response_attempt_{attempt}", {
                "question": question,
                "attempt": attempt,
                "raw_response": content,
            })
            parsed = extract_json_block(content)
            validation_error = module2_validation_error(parsed)
            if debug_hook is not None:
                debug_hook({
                    "attempt": attempt,
                    "raw": content,
                    "parsed": parsed,
                    "validation_error": validation_error,
                })

            if not validation_error:
                if parsed and isinstance(parsed, dict) and "output" in parsed:
                    parsed = parsed["output"]

                if parsed:
                    parsed = force_numeric_answer_type(parsed)
                    log_json(f"{trace_id}_module2_parsed_attempt_{attempt}", parsed)
                return parsed

            repaired = repair_module2_schema(parsed, question)
            if repaired:
                print("  Module 2: repaired JSON schema", flush=True)
                return repaired

            print(f"  Module 2: invalid schema on attempt {attempt}", flush=True)

        except Exception as exc:
            print(f"  Module 2: error on attempt {attempt}: {exc}", flush=True)

        time.sleep(runtime_config.SLEEP_SECONDS)

    fallback = fallback_module2_from_question(question)
    if fallback:
        fallback = force_numeric_answer_type(fallback)
    if fallback and validate_module2(fallback):
        print("  Module 2: using deterministic fallback extraction", flush=True)
        return fallback
    return None


def _trace_id_from_question(question: str) -> str:
    m = re.search(r"\b([A-Z]{2,5}\d{3})\b", str(question or ""))
    if m:
        return m.group(1)
    digest = hashlib.sha1(str(question or "").encode("utf-8")).hexdigest()[:10]
    return f"question_{digest}"
