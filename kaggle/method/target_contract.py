"""Target-type inference and lightweight output contract for legacy SymPlanner."""

from __future__ import annotations

import json
import re
import ast
from typing import Any, Mapping

ANSWER_TYPES = {"number", "symbolic", "tuple", "set", "matrix", "text", "base_notation"}
PLANNER_TYPE_ALIASES = {
    "complex_number": "number",
    "complex": "number",
    "common fraction": "number",
    "fraction": "number",
    "integer": "number",
    "polynomial": "symbolic",
    "expression": "symbolic",
    "base notation": "base_notation",
}


def _explicit_planner_answer_type(planner_note: str) -> str | None:
    text = str(planner_note or "")
    match = re.search(r'["\']answer_type["\']\s*:\s*["\']([^"\']+)["\']', text)
    if not match:
        match = re.search(r"^\s*#\s*Answer\s+type\s*:\s*([^\r\n]+)", text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    answer_type = match.group(1).strip().lower().replace("-", "_")
    answer_type = PLANNER_TYPE_ALIASES.get(answer_type, answer_type)
    return answer_type if answer_type in ANSWER_TYPES else None


def infer_target_spec(question: str, planner_note: str = "") -> dict[str, Any]:
    """Infer the output shape from wording without attempting to solve the problem."""
    text = str(question or "").lower()
    planner_answer_type = _explicit_planner_answer_type(planner_note)
    spec: dict[str, Any] = {
        "answer_type": "number",
        "unit": None,
        "diagram_required": bool(re.search(r"\[(?:asy|asy\s*\n)|tikzpicture|begin\{picture\}", text)),
        "target_count": 1,
    }
    asks_numeric_value = bool(re.search(r"\b(?:find|what is|compute|determine)\b[^?.]*\b(?:value|sum|product|difference|minimum|maximum|smallest|largest)\b", text))
    asks_text_entity = bool(re.search(
        r"\bwho\b|\bname of\b|\bwhich\s+(?:student|person|team|city|country|runner|contestant|player)\b",
        text,
    ))
    if asks_text_entity and not re.search(r"\bfor which\b", text):
        spec["answer_type"] = "text"
    elif re.search(r"\b(even|odd|neither)\b", text) or "true or false" in text:
        spec["answer_type"] = "text"
    elif re.search(r"\bin terms of\b|\bexpress .* using\b|\bpolynomial in\b|\bfunction .* of\b", text):
        spec["answer_type"] = "symbolic"
    elif re.search(r"\b(simplify|expand|factor)\b", text) and re.search(r"\b[a-z]\b", text):
        spec["answer_type"] = "symbolic"
    elif re.search(
        r"\bfind\s+(?:all\s+|the\s+)?(?:values|roots|solutions|zeros)\b|\benter\s+all\b|\bwhat\s+are\s+(?:the\s+)?(?:roots|solutions|zeros)\b",
        text,
    ):
        spec["answer_type"] = "set"
    elif re.search(r"\b(matrix|determinant|eigenvalue)\b", text):
        spec["answer_type"] = "matrix"
    elif not asks_numeric_value and re.search(r"\b(polar coordinates?|ordered pair|ordered triple|ordered quadruple|coordinate pair|coordinates of|coordinates of the point|find the point|what point)\b", text):
        spec["answer_type"] = "tuple"
    elif re.search(r"\b(base|binary|octal|hexadecimal|base-\d+)\b", text) and re.search(r"\b(write|express|convert|in base)\b", text):
        spec["answer_type"] = "base_notation"
    elif planner_answer_type is not None:
        spec["answer_type"] = planner_answer_type
    if re.search(r"\bpercent(age)?\b|\bprobability\b", text):
        spec["unit"] = "%" if "percent" in text or "percentage" in text else None
    return spec


def _parse_labeled_planner(text: str) -> dict[str, Any] | None:
    """Parse the compact line format used by the <=8B planner."""
    has_label = re.search(
        r"^\s*#\s*(?:Subject|Target|Given|Step\s+\d+|Answer\s+type)\s*:",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not has_label:
        return None

    def field(label: str) -> str:
        match = re.search(
            rf"^\s*#\s*{label}\s*:\s*([^\r\n]*)",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        return match.group(1).strip() if match else ""

    given_text = field("Given")
    given = [part.strip() for part in given_text.split(";") if part.strip()] or ([given_text] if given_text else [])
    steps = [
        match.group(1).strip()
        for match in re.finditer(
            r"^\s*#\s*Step\s+\d+\s*:\s*([^\r\n]*)",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if match.group(1).strip()
    ]
    answer_type = field(r"Answer\s+type").lower().replace("-", "_")
    answer_type = PLANNER_TYPE_ALIASES.get(answer_type, answer_type)
    return {
        "subject": field("Subject").lower(),
        "target_unknown": field("Target"),
        "given_constants": given,
        "strategy": "",
        "steps": steps,
        "pitfalls": [],
        "answer_type": answer_type,
    }


def parse_planner_contract(raw_plan: str, question: str = "") -> tuple[str, dict[str, Any], list[str]]:
    """Parse compact labeled planner output, with backward-compatible JSON support."""
    text = str(raw_plan or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if "<think>" in text:
        text = text.split("<think>", 1)[0].strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text
    errors: list[str] = []
    parsed: dict[str, Any] = {}
    labeled = _parse_labeled_planner(candidate)
    if labeled is not None:
        parsed = labeled
    else:
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
    answer_type = str(parsed.get("answer_type", "")).strip().lower().replace("-", "_")
    answer_type = PLANNER_TYPE_ALIASES.get(answer_type, answer_type)
    if answer_type in ANSWER_TYPES:
        parsed["answer_type"] = answer_type
    else:
        parsed["answer_type"] = inferred["answer_type"]
    for key in ("given_constants", "steps", "pitfalls"):
        if key in parsed and not isinstance(parsed[key], list):
            errors.append(f"planner {key} must be a list")
    return candidate[:1500].strip(), parsed, list(dict.fromkeys(errors))


def target_contract_feedback(question: str, candidate_answer: Any, planner_note: str = "") -> tuple[str, str] | None:
    """Return a verifier result when the candidate visibly violates the target type."""
    spec = infer_target_spec(question, planner_note)
    answer = str(candidate_answer or "").strip()
    q_lower = str(question or "").lower()
    if not answer:
        return "fail", "Verification Error: empty candidate answer."
    if spec["answer_type"] == "text":
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\s*[A-Za-z]+)?", answer):
            return "fail", "Verification Error: target requires a named text entity, but candidate is numeric."
        return "unknown", f"Candidate answer is a valid text entity ('{answer}')."
    elif spec["answer_type"] == "symbolic":
        explicitly_symbolic = re.search(r"\bin terms of\b|\bexpress .* using\b|\bpolynomial in\b|\bfunction .* of\b", q_lower)
        if explicitly_symbolic and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer):
            return "fail", "Verification Error: target must remain symbolic in the requested parameters."
    elif spec["answer_type"] == "set":
        if not (answer.startswith(("[", "{")) or "," in answer):
            return "unknown", "Verification Unknown: scalar answer is possible for a single-solution set request."
    elif spec["answer_type"] == "tuple":
        if not ((answer.startswith("(") and answer.endswith(")")) or (answer.startswith("[") and answer.endswith("]"))):
            return "fail", "Verification Error: target requires a coordinate tuple."
    elif spec["answer_type"] == "base_notation":
        if not re.search(r"[_\\]", answer):
            return "fail", "Verification Error: preserve the requested base notation in the answer."
    if spec["diagram_required"] and spec["answer_type"] == "number":
        return "unknown", "Verification Unknown: diagram-dependent target needs relation-level evidence."
    return None


def _as_numeric_list(answer: Any) -> list[float] | None:
    if isinstance(answer, (list, tuple)):
        values = list(answer)
    else:
        text = str(answer or "").strip()
        if not (text.startswith(("[", "(")) and text.endswith(("]", ")"))):
            return None
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return None
        if not isinstance(parsed, (list, tuple)):
            return None
        values = list(parsed)
    numbers = []
    for value in values:
        try:
            numbers.append(float(value))
        except Exception:
            return None
    return numbers


def format_answer_for_contract(question: str, answer: Any, answer_type: str | None = None) -> Any:
    """Repair display-only answer notation required by the prompt target."""
    if answer is None:
        return answer
    spec = infer_target_spec(question)
    inferred_type = spec.get("answer_type")
    answer_text = str(answer).strip()
    numeric_list = _as_numeric_list(answer)
    if inferred_type == "tuple" and numeric_list and len(numeric_list) >= 2:
        parts = [str(int(value)) if value.is_integer() else str(value) for value in numeric_list]
        return "(" + ",".join(parts) + ")"
    if inferred_type == "number" and numeric_list and re.search(r"\b(?:x\s*\+\s*y|a\s*\+\s*b|p\s*\+\s*q)\b", str(question or ""), flags=re.IGNORECASE):
        total = sum(numeric_list)
        return str(int(total)) if total.is_integer() else str(total)
    if inferred_type == "base_notation" and answer_text and not re.search(r"[_\\]", answer_text):
        base_match = re.search(r"\bbase\s*\$?(\d+)\$?|\bin\s+base\s*\$?(\d+)\$?", str(question or ""), flags=re.IGNORECASE)
        if base_match:
            base = next(group for group in base_match.groups() if group)
            return f"{answer_text}_{base}"
    return answer
