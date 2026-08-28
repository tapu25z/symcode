"""Reference orchestration for the new method.

The class is intentionally dependency-injected: existing model and sandbox
implementations can be plugged in without coupling this experiment to one SDK.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Mapping

from .config import AblationConfig, resolve_ablation
from .normalizer import build_codegen_payload, normalize_problem_ir, validate_normalized_ir
from .problem_ir import ALLOWED_OUTPUT_TYPES, extract_json_object, normalize_ir_shape, validate_ir
from .prompts import codegen_prompt, extraction_prompt, ir_repair_prompt, lean_codegen_prompt, lean_repair_prompt, repair_prompt
from .relation_verifier import validate_execution_output, verify_bidirectional


def strip_code_fence(text: str) -> str:
    """Accept plain Python or a markdown-fenced model response."""
    value = (text or "").strip()
    matches = re.findall(r"```(?:python|py)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


class SymPlannerIRPipeline:
    def __init__(
        self,
        llm_call: Callable[[list[dict[str, str]]], str],
        execute_code: Callable[[str], Mapping[str, Any]],
        max_repairs: int = 2,
        max_ir_repairs: int = 1,
        ablation: str | AblationConfig | None = None,
        answer_verifier: Callable[[str, Any, str | None, str | None], tuple[str, str]] | None = None,
    ):
        self.llm_call = llm_call
        self.execute_code = execute_code
        self.max_repairs = max_repairs
        self.max_ir_repairs = max_ir_repairs
        self.ablation = resolve_ablation(ablation)
        self.answer_verifier = answer_verifier

    def run(self, question: str) -> dict[str, Any]:
        if self.ablation.pipeline == "lean":
            return self._run_lean(question)
        return self._run_ir(question)

    def _run_ir(self, question: str) -> dict[str, Any]:
        required_keys = {"target_unknown", "givens", "relations", "conditions", "required_output"}

        def collect_schema_errors(raw: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
            errors = [] if raw else ["extractor returned no valid JSON"]
            errors.extend(f"missing top-level key: {key}" for key in sorted(required_keys - set(raw)))
            errors.extend(f"unexpected top-level key: {key}" for key in sorted(set(raw) - required_keys))
            errors.extend(validate_ir(candidate))
            return list(dict.fromkeys(errors))

        raw_ir = extract_json_object(self.llm_call(extraction_prompt(question)))
        ir = normalize_ir_shape(raw_ir)
        schema_errors = collect_schema_errors(raw_ir, ir)
        ir_repairs = []
        ir_repair_budget = self.max_ir_repairs if self.ablation.enable_ir_repair else 0
        for attempt in range(ir_repair_budget):
            if not schema_errors:
                break
            repaired = extract_json_object(self.llm_call(ir_repair_prompt(question, ir, schema_errors)))
            ir_repairs.append({"attempt": attempt + 1, "errors": schema_errors, "candidate": repaired})
            if not repaired:
                break
            raw_ir, ir = repaired, normalize_ir_shape(repaired)
            schema_errors = collect_schema_errors(raw_ir, ir)
        if schema_errors:
            return {"status": "invalid_ir", "variant": self.ablation.name, "question": question, "ir": ir, "payload": None, "code": None, "schema_errors": schema_errors, "ir_repairs": ir_repairs, "attempts": [], "final": {"status": "invalid_ir", "errors": schema_errors}}
        normalized = normalize_problem_ir(ir)
        normalization_errors = validate_normalized_ir(normalized)
        if normalization_errors:
            return {"status": "invalid_ir", "variant": self.ablation.name, "question": question, "ir": normalized, "payload": None, "code": None, "schema_errors": normalization_errors, "ir_repairs": ir_repairs, "attempts": [], "final": {"status": "invalid_ir", "errors": normalization_errors}}
        payload = build_codegen_payload(normalized)
        code = strip_code_fence(self.llm_call(codegen_prompt(payload)))
        attempts = []
        code_repair_budget = self.max_repairs if self.ablation.enable_code_repair else 0
        for attempt in range(code_repair_budget + 1):
            try:
                execution_result = self.execute_code(code)
                execution = dict(execution_result or {})
            except Exception as exc:
                execution = {"execution_error": f"{type(exc).__name__}: {exc}"}
            if self.ablation.enable_bidirectional_verifier:
                verification = verify_bidirectional(normalized, execution)
            else:
                output_errors = validate_execution_output(execution, normalized)
                verification = {"status": "fail" if output_errors else "pass", "checks": [], "failures": [], "unknown": [], "output_errors": output_errors, "feedback": [f"Output contract error: {error}" for error in output_errors]}
            attempts.append({"attempt": attempt, "execution": execution, "verification": verification})
            if verification["status"] == "pass":
                break
            if attempt >= code_repair_budget:
                break
            code = strip_code_fence(self.llm_call(repair_prompt(payload, code, {"execution": execution, "verification": verification})))
        final_status = attempts[-1]["verification"]["status"] if attempts else "fail"
        return {"status": final_status, "variant": self.ablation.name, "question": question, "ir": normalized, "payload": payload, "code": code, "schema_errors": schema_errors, "ir_repairs": ir_repairs, "attempts": attempts, "final": attempts[-1] if attempts else None}

    def _run_lean(self, question: str) -> dict[str, Any]:
        code = strip_code_fence(self.llm_call(lean_codegen_prompt(question)))
        attempts = []
        code_repair_budget = self.max_repairs if self.ablation.enable_code_repair else 0
        for attempt in range(code_repair_budget + 1):
            try:
                execution_result = self.execute_code(code)
                execution = dict(execution_result or {})
            except Exception as exc:
                execution = {"execution_error": f"{type(exc).__name__}: {exc}"}
            verification = self._verify_lean_output(question, code, execution)
            attempts.append({"attempt": attempt, "execution": execution, "verification": verification})
            if verification["status"] == "pass":
                break
            if attempt >= code_repair_budget:
                break
            code = strip_code_fence(self.llm_call(lean_repair_prompt(question, code, {"execution": execution, "verification": verification})))
        final_status = attempts[-1]["verification"]["status"] if attempts else "fail"
        return {
            "status": final_status,
            "variant": self.ablation.name,
            "question": question,
            "ir": None,
            "payload": {"problem": question, "mode": "lean_problem_to_code"},
            "code": code,
            "schema_errors": [],
            "ir_repairs": [],
            "attempts": attempts,
            "final": attempts[-1] if attempts else None,
        }

    def _verify_lean_output(self, question: str, code: str, execution: Mapping[str, Any]) -> dict[str, Any]:
        output_errors = validate_lean_execution_output(execution)
        feedback = [f"Output contract error: {error}" for error in output_errors]
        verifier_status = "not_applicable"
        verifier_feedback = None
        if not output_errors and self.answer_verifier:
            verifier_status, verifier_feedback = self.answer_verifier(
                question,
                execution.get("answer"),
                code,
                execution.get("_stdout") if isinstance(execution.get("_stdout"), str) else None,
            )
            if verifier_feedback:
                feedback.append(verifier_feedback)
        status = "fail" if output_errors or verifier_status == "fail" else "pass"
        return {
            "status": status,
            "checks": [],
            "failures": [],
            "unknown": [],
            "output_errors": output_errors,
            "verifier_status": verifier_status,
            "feedback": feedback,
        }


def validate_lean_execution_output(execution: Mapping[str, Any]) -> list[str]:
    runtime_errors = [str(execution[key]) for key in ("error", "execution_error", "stderr") if execution.get(key)]
    if runtime_errors:
        return [f"execution error: {item}" for item in runtime_errors]

    errors: list[str] = []
    required_fields = ("answer", "canonical_answer", "answer_type", "unit", "variables")
    for field in required_fields:
        if field not in execution:
            errors.append(f"execution output missing field: {field}")

    answer_type = str(execution.get("answer_type") or "")
    if answer_type not in ALLOWED_OUTPUT_TYPES:
        errors.append(f"execution.answer_type must be one of {sorted(ALLOWED_OUTPUT_TYPES)}")

    for field in ("answer", "canonical_answer"):
        value = execution.get(field)
        text = str(value or "").strip()
        if not text:
            errors.append(f"execution.{field} must be non-empty")
        elif text.lower() in {"none", "null", "nan", "inf", "infinity", "-inf", "-infinity", "undefined", "invalid"}:
            errors.append(f"execution.{field} is not a concrete answer")

    variables = execution.get("variables")
    if not isinstance(variables, Mapping):
        errors.append("execution.variables must be an object")
    else:
        for symbol, value in variables.items():
            if not isinstance(symbol, str) or not symbol.isidentifier():
                errors.append(f"execution.variables has invalid symbol: {symbol}")
            if isinstance(value, float) and not math.isfinite(value):
                errors.append(f"execution.variables[{symbol}] must be finite")

    for field in ("answer", "canonical_answer"):
        value = execution.get(field)
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"execution.{field} must be finite")

    return errors
