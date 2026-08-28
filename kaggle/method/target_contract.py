"""Target-type inference and lightweight output contract for legacy SymPlanner."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


def infer_target_spec(question: str, planner_note: str = "") -> dict[str, Any]:
    """Infer the output shape from wording without attempting to solve the problem."""
    text = f"{question}\n{planner_note}".lower()
    spec: dict[str, Any] = {
        "answer_type": "number",
        "unit": None,
        "diagram_required": bool(re.search(r"\[(?:asy|asy\s*\n)|tikzpicture|begin\{picture\}", text)),
        "target_count": 1,
    }
    if re.search(r"\b(which|who|name of|student|person|team|city|country)\b", text) and re.search(r"\b(which|who|name of)\b", text):
        spec["answer_type"] = "text"
    elif re.search(r"\bin terms of\b|\bexpress .* using\b|\bpolynomial in\b|\bfunction .* of\b", text):
        spec["answer_type"] = "symbolic"
    elif re.search(r"\b(find all|all values|roots|solutions|zeros)\b", text):
        spec["answer_type"] = "set"
    elif re.search(r"\b(matrix|determinant|eigenvalue)\b", text):
        spec["answer_type"] = "matrix"
    elif re.search(r"\b(polar coordinates?|ordered pair|coordinate pair|coordinates of)\b", text):
        spec["answer_type"] = "tuple"
    elif re.search(r"\b(base|binary|octal|hexadecimal|base-\d+)\b", text) and re.search(r"\b(write|express|convert|in base)\b", text):
        spec["answer_type"] = "base_notation"
    if re.search(r"\bpercent(age)?\b|\bprobability\b", text):
        spec["unit"] = "%" if "percent" in text or "percentage" in text else None
    return spec


def parse_planner_contract(raw_plan: str, question: str = "") -> tuple[str, dict[str, Any], list[str]]:
    """Parse planner JSON and repair only safe missing metadata locally."""
    text = str(raw_plan or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text
    errors: list[str] = []
    parsed: dict[str, Any] = {}
    try:
        value = json.loads(candidate)
        if isinstance(value, Mapping):
            parsed = dict(value)
        else:
            errors.append("planner output must be a JSON object")
    except json.JSONDecodeError as exc:
        errors.append(f"planner JSON parse failed: {exc.msg}")
    required = ("target_unknown", "given_constants", "strategy", "steps", "pitfalls", "answer_type")
    errors.extend(f"planner missing key: {key}" for key in required if key not in parsed)
    inferred = infer_target_spec(question, candidate)
    if parsed.get("answer_type") not in {"number", "symbolic", "tuple", "set", "matrix", "text", "base_notation"}:
        parsed["answer_type"] = inferred["answer_type"]
        if "answer_type" not in parsed:
            errors.append("planner answer_type is missing")
    for key in ("given_constants", "steps", "pitfalls"):
        if key in parsed and not isinstance(parsed[key], list):
            errors.append(f"planner {key} must be a list")
    return candidate[:1500].strip(), parsed, list(dict.fromkeys(errors))


def target_contract_feedback(question: str, candidate_answer: Any, planner_note: str = "") -> tuple[str, str] | None:
    """Return a verifier result when the candidate visibly violates the target type."""
    spec = infer_target_spec(question, planner_note)
    answer = str(candidate_answer or "").strip()
    if not answer:
        return "fail", "Verification Error: empty candidate answer."
    if spec["answer_type"] == "text":
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\s*[A-Za-z]+)?", answer):
            return "fail", "Verification Error: target requires a named text entity, but candidate is numeric."
    elif spec["answer_type"] == "symbolic":
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer):
            return "fail", "Verification Error: target must remain symbolic in the requested parameters."
    elif spec["answer_type"] == "set":
        if not (answer.startswith(("[", "{")) or "," in answer):
            return "fail", "Verification Error: target requires a set/list of all requested solutions."
    elif spec["answer_type"] == "tuple":
        if not (answer.startswith("(") and answer.endswith(")")):
            return "fail", "Verification Error: target requires a coordinate tuple."
    elif spec["answer_type"] == "base_notation":
        if not re.search(r"[_\\]", answer):
            return "fail", "Verification Error: preserve the requested base notation in the answer."
    if spec["diagram_required"] and spec["answer_type"] == "number":
        return "unknown", "Verification Unknown: diagram-dependent target needs relation-level evidence."
    return None
