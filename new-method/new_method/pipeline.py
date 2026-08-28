"""Single IR pipeline: extract, normalize, plan/codegen, execute and verify."""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping

from .config import AblationConfig, resolve_ablation
from .normalizer import augment_ir_from_question, build_codegen_payload, normalize_problem_ir, validate_normalized_ir
from .problem_ir import extract_json_object, normalize_ir_shape, validate_ir
from .prompts import codegen_prompt, extraction_prompt, repair_prompt
from .relation_verifier import verify_bidirectional


RUNTIME_HEADER = r"""import sympy as sp
import math
import json
from fractions import Fraction

def enc(v):
    if isinstance(v, bool):
        return v
    if getattr(v, "is_Integer", False):
        return int(v)
    if isinstance(v, float):
        return int(v) if v.is_integer() else v
    return str(v) if isinstance(v, sp.Basic) else v

def safe_eval(v):
    return v if isinstance(v, sp.Basic) else sp.sympify(v)
"""


def strip_code_fence(text: str) -> str:
    """Accept plain Python or a markdown-fenced model response."""
    value = (text or "").strip()
    matches = re.findall(r"```(?:python|py)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return value


class SymPlannerIRPipeline:
    """Run exactly one extraction and one plan/codegen call.

    IR errors are retained as diagnostics. Only an unusable target/relation
    graph stops the pipeline; execution and verifier failures use code repair.
    """

    def __init__(
        self,
        llm_call: Callable[[list[dict[str, str]]], str],
        execute_code: Callable[[str], Mapping[str, Any]],
        max_repairs: int = 2,
        ablation: str | AblationConfig | None = None,
    ):
        self.llm_call = llm_call
        self.execute_code = execute_code
        self.max_repairs = min(max(0, int(max_repairs)), 2)
        self.ablation = resolve_ablation(ablation)

    def run(self, question: str) -> dict[str, Any]:
        raw_ir = extract_json_object(self.llm_call(extraction_prompt(question)))
        ir = normalize_ir_shape(raw_ir)
        schema_errors = self._schema_errors(raw_ir, ir)
        normalized = augment_ir_from_question(question, normalize_problem_ir(ir))
        normalization_errors = validate_normalized_ir(normalized)
        all_ir_errors = list(dict.fromkeys(schema_errors + normalization_errors))

        fatal_errors = self._fatal_ir_errors(ir, all_ir_errors)
        if fatal_errors:
            return self._invalid_ir(question, normalized, all_ir_errors, fatal_errors)

        payload = build_codegen_payload(normalized)
        code = strip_code_fence(self.llm_call(codegen_prompt(payload)))
        attempts: list[dict[str, Any]] = []
        for attempt in range(self.max_repairs + 1):
            execution = self._execute(code)
            verification = verify_bidirectional(normalized, execution)
            attempts.append({"attempt": attempt, "execution": execution, "verification": verification})
            if verification["status"] == "pass" or not self._repair_would_help(verification) or attempt >= self.max_repairs:
                break
            code = strip_code_fence(self.llm_call(repair_prompt(payload, code, verification)))

        final = attempts[-1] if attempts else None
        return {
            "status": final["verification"]["status"] if final else "fail",
            "variant": self.ablation.name,
            "question": question,
            "ir": normalized,
            "payload": payload,
            "code": code,
            "schema_errors": all_ir_errors,
            "ir_repairs": [],
            "attempts": attempts,
            "final": final,
        }

    def _schema_errors(self, raw_ir: Mapping[str, Any], ir: Mapping[str, Any]) -> list[str]:
        errors = [] if raw_ir else ["extractor returned no valid JSON"]
        errors.extend(validate_ir(ir))
        return list(dict.fromkeys(errors))

    @staticmethod
    def _fatal_ir_errors(
        ir: Mapping[str, Any], errors: list[str]
    ) -> list[str]:
        """Keep schema diagnostics, but reserve invalid_ir for unusable IR."""
        fatal: list[str] = []
        if not ir.get("relations"):
            fatal.append("at least one usable relation is required")
        target = ir.get("target_unknown")
        if not isinstance(target, Mapping) or not str(target.get("symbol") or "").strip():
            fatal.append("target_unknown.symbol is required")
        output = ir.get("required_output")
        if not isinstance(output, Mapping) or not output.get("type"):
            fatal.append("required_output.type is required")
        for error in errors:
            if any(marker in error for marker in (
                "must be a non-empty expression",
                "target_count must equal",
                "has unsupported operator",
            )):
                fatal.append(error)
        return list(dict.fromkeys(fatal))

    @staticmethod
    def _invalid_ir(
        question: str,
        normalized: Mapping[str, Any],
        diagnostics: list[str],
        fatal_errors: list[str],
    ) -> dict[str, Any]:
        return {
            "status": "invalid_ir",
            "variant": "IR",
            "question": question,
            "ir": normalized,
            "payload": None,
            "code": None,
            "schema_errors": diagnostics,
            "ir_repairs": [],
            "attempts": [],
            "final": {"status": "invalid_ir", "errors": fatal_errors},
        }

    def _execute(self, code: str) -> dict[str, Any]:
        try:
            return dict(self.execute_code(f"{RUNTIME_HEADER}\n{code}") or {})
        except Exception as exc:
            return {"execution_error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _repair_would_help(verification: Mapping[str, Any]) -> bool:
        """Avoid spending repairs on merely unverifiable auxiliary relations."""
        if verification.get("status") != "unknown":
            return True
        if verification.get("output_errors") or verification.get("failures"):
            return True
        feedback = verification.get("feedback") or []
        return any("Missing canonical target answer" in str(item) for item in feedback)
