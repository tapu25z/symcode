"""
Định nghĩa hệ thống prompt và bộ tạo thông điệp ChatML cho các phương pháp benchmark:
Direct, CoT, SymCode và SymPlanner (Decoupled Divide-and-Plan Pipeline: Planner -> Pure Codegen -> Debug Repair).
"""

import re
import json
from typing import Dict, List, Optional, Any

from .target_contract import infer_target_spec
from .problem_hints import build_problem_hints

# ==============================================================================
# 1. PLANNER PROMPTS (Turn 1: Phân tích & Lập kế hoạch ngắn gọn, không sinh code)
# ==============================================================================

PLANNER_SYSTEM_PROMPT = r"""You are a careful mathematical planner.

Think through the problem, then output a compact structured plan that will be used by a separate code generator.
The plan must be short, practical, and precise. Do NOT generate Python code.

Output ONLY this JSON object:
{
  "target_unknown": "exact quantity or simplified expression to find",
  "given_constants": ["key numbers, parameters and relations from problem"],
  "strategy": "sequential arithmetic OR symbolic equation solving (sp.solve)",
  "steps": ["step 1", "step 2", "step 3"],
  "pitfalls": ["unit/sign/rounding/free-variable pitfalls to avoid"],
  "answer_type": "number|symbolic|tuple|set|matrix|text|base_notation"
}

Keep the final JSON under 120 words."""

# ==============================================================================
# 2. CODEGEN PROMPTS (Turn 2: Sinh mã nguồn Python/SymPy thuần túy 100%)
# ==============================================================================

SYMPLANNER_CODEGEN_SYSTEM_PROMPT = r"""You are an expert mathematical solver and deterministic Python/SymPy code generator.

Given one math problem and planner notes, return ONLY executable Python code enclosed in a single ```python ... ``` block.
Do NOT write explanations. Do NOT output markdown text outside the code fence. Do NOT output <think> tags.

The code MUST:
1. import sympy as sp and import json (plus math, fractions, itertools if helpful).
2. Define given values and variables clearly.
3. Compute the requested target quantity:
   - For sequential word problems (GSM8K style): Compute step-by-step using Python/SymPy arithmetic.
   - For algebraic systems (MATH style): Use sp.symbols(...) with domain assumptions and sp.solve(...).
4. CRITICAL RULES:
   - Never output `None`, `Invalid`, or undefined variables.
   - Do NOT create conditional `if/else` checks that assign `None` or `Invalid`.
   - Never call `.evalf()` on Python standard `int` or `float`.
   - Prefer exact SymPy/Rational arithmetic. Do not convert to float unless the problem explicitly asks for a decimal approximation.
   - Always solve for the requested target in the required output type; never print an intermediate quantity.
   - Respect the target output type inferred from the question: text/entity names must not be replaced by a numeric score; symbolic targets must preserve named parameters; sets must include all solutions; tuples must contain all coordinates; base notation must be preserved.
   - Never silently replace a diagram-dependent quantity with an arbitrary numeric guess. If the diagram is required, encode its stated coordinates/relations explicitly.
   - Avoid using sp.solve() or sp.nonlinsolve() on complex nonlinear or multivariate systems of high degree (e.g. degree >= 3 with multiple variables, or equations containing non-rational exponent powers like **(1/3)), as it causes SymPy to hang indefinitely. Use numerical optimization (e.g., scipy.optimize.minimize or fsolve) instead.
   - Never write infinite loops or unbounded while loops (e.g., custom prime generators). Always use finite for loops (e.g., for i in range(10000)) or specify a maximum iteration count to guarantee termination.
   - Read and strictly apply the # PROBLEM-SPECIFIC IMPLEMENTATION HINTS. They contain exact math models, safe SymPy API formulas, or search procedures required for this specific problem.
   - Double check all mathematical operators (+, -, *, /) in the prompt against your generated code. For example, if the problem subtracts two fractions, write a minus sign (-), not a plus sign (+).
5. Before printing, add cheap internal checks whenever possible:
   - Substitute candidate solutions back into equations/inequalities.
   - For small combinatorics, brute-force enumerate and compare against any formula.
   - For symbolic identities, expand/simplify both sides at exact or multiple sample values.
   - For optimization/norm problems, compare analytic result against numeric samples.
6. Print exactly one JSON line and nothing else at the very end:
   print(json.dumps({"answer": str(display_answer), "canonical_answer": str(canonical_answer), "answer_type": "<target type>", "unit": None, "variables": {}}, default=str))
   Do not use `.format(...)` or an f-string to hand-build JSON; braces in JSON conflict with those formatting methods.
   `answer_type` must match the OUTPUT CONTRACT.
"""

SYMCODE_SYSTEM_PROMPT = r"""You are an expert mathematical solver and deterministic Python/SymPy code generator.

Solve the problem by returning ONLY executable Python code enclosed in a single ```python ... ``` block.
Do NOT write explanations. Do NOT output <think> tags.

The code MUST:
1. import sympy as sp (and math, fractions if helpful).
2. Define all given quantities and formulate equations accurately.
3. Solve for the target quantity symbolically or numerically.
4. Never call `.evalf()` on standard Python int/float.
5. Avoid using sp.solve() or sp.nonlinsolve() on complex nonlinear or multivariate systems of high degree (e.g. degree >= 3 with multiple variables, or equations containing non-rational exponent powers like **(1/3)), as it causes SymPy to hang indefinitely. Use numerical optimization (e.g., scipy.optimize.minimize or fsolve) instead.
6. Never write infinite loops or unbounded while loops (e.g., custom prime generators). Always use finite for loops (e.g., for i in range(10000)) or specify a maximum iteration count to guarantee termination.
7. Print ONLY the final answer in LaTeX boxed format at the end:
   print(f"\\boxed{{{final_answer}}}")
"""

# ==============================================================================
# 3. DEBUG / REPAIR PROMPTS (Turn 3: Sửa lỗi mã nguồn có chủ đích)
# ==============================================================================

DEBUG_SYSTEM_PROMPT = r"""You are fixing Python/SymPy code for a math problem.

Return ONLY corrected executable Python code enclosed in a single ```python ... ``` block.
Do NOT write explanations. Do NOT output <think> tags.

Fix the shown failure:
- Syntax error / Indentation error
- Runtime error / AttributeError (e.g. calling .evalf() on int/float)
- Free variables remaining in answer (e.g., answer contains symbols like '48 - 6*z')
- Empty stdout or wrong output format
- Answer evaluated to None / Invalid

Requirements:
1. Ensure the code computes a concrete numerical value or simplified expression.
2. Avoid using sp.solve() or sp.nonlinsolve() on complex nonlinear or multivariate systems of high degree (e.g. degree >= 3 with multiple variables, or equations containing non-rational exponent powers like **(1/3)), as it causes SymPy to hang indefinitely. Use numerical optimization (e.g., scipy.optimize.minimize or fsolve) instead.
3. Never write infinite loops or unbounded while loops. Always use finite for loops (e.g., for i in range(10000)) or specify a maximum iteration count to guarantee termination.
4. Print ONLY the final answer in LaTeX boxed format:
   print(f"\\boxed{{{final_answer}}}")
"""

SYMPLANNER_DEBUG_SYSTEM_PROMPT = r"""You are fixing Python/SymPy code for a math problem.

Return ONLY corrected executable Python code enclosed in a single ```python ... ``` block.
Do NOT write explanations or <think> tags.

Fix the shown execution, contract, or verifier failure. Recompute the answer from
the problem; do not hard-code a replacement. Print exactly one JSON line with
exactly answer, canonical_answer, answer_type, unit, and variables. Keep the
answer_type required by the OUTPUT CONTRACT and do not print debug output. Use
print(json.dumps(..., default=str)); do not hand-format JSON with .format or an
f-string.

CRITICAL RULES:
- Avoid using sp.solve() or sp.nonlinsolve() on complex nonlinear or multivariate systems of high degree (e.g. degree >= 3 with multiple variables, or equations containing non-rational exponent powers like **(1/3)), as it causes SymPy to hang indefinitely. Use numerical optimization (e.g., scipy.optimize.minimize or fsolve) instead.
- Never write infinite loops or unbounded while loops. Always use finite for loops (e.g., for i in range(10000)) or specify a maximum iteration count to guarantee termination.
- Read and strictly apply the # PROBLEM-SPECIFIC IMPLEMENTATION HINTS provided in the context to construct a correct solution.
"""

# ==============================================================================
# 4. BASELINE PROMPTS (Direct & CoT)
# ==============================================================================

COT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem step-by-step with clear and rigorous logical reasoning.
At the end of your reasoning, write your final answer strictly formatted in \\boxed{answer}."""

DIRECT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem directly.
Do not provide long explanations. Put only the final answer inside \\boxed{answer}."""

SYSTEM_PROMPTS = {
    "Direct": DIRECT_SYSTEM_PROMPT,
    "CoT": COT_SYSTEM_PROMPT,
    "SymCode": SYMCODE_SYSTEM_PROMPT,
    "SymPlanner": SYMPLANNER_CODEGEN_SYSTEM_PROMPT,
}


# ==============================================================================
# 5. HELPER FUNCTIONS & MESSAGE BUILDERS
# ==============================================================================

def remove_thinking_tags(text: str) -> str:
    """Loại bỏ các thẻ <think>...</think> của các mô hình reasoning (Qwen, DeepSeek...)."""
    text = str(text or "")
    if "<think>" in text and "</think>" not in text:
        return text.split("<think>", 1)[0].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text.strip()


def clean_planner_note(raw_plan: str) -> str:
    """
    Làm sạch kết quả kế hoạch từ Turn 1:
    - Loại bỏ thẻ thinking.
    - Trích xuất khối JSON hoặc văn bản kế hoạch có giới hạn độ dài để không làm phình context codegen.
    """
    if not raw_plan or not raw_plan.strip():
        return ""
    text = remove_thinking_tags(raw_plan.strip())
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return text[:1500].strip()


def build_planner_messages(question: str) -> List[Dict[str, str]]:
    """Xây dựng thông điệp cho Turn 1: Lập kế hoạch phân tích đề bài."""
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n\nThink if needed, then output the compact JSON plan only. Do not write code."}
    ]


def build_symplanner_codegen_messages(question: str, planner_note: str = "") -> List[Dict[str, str]]:
    """Xây dựng thông điệp cho Turn 2: Sinh mã nguồn Python/SymPy từ đề bài + Kế hoạch."""
    plan_block = planner_note.strip() or "Planner note unavailable. Solve directly from first principles."
    target_spec = infer_target_spec(question, planner_note)
    target_block = json.dumps(target_spec, sort_keys=True)
    hints = build_problem_hints(question)
    hints_block = "\n".join(f"- {hint}" for hint in hints) if hints else "- No extra pattern hint."
    user_content = f"""# PROBLEM
{question}

# PLANNER NOTES
{plan_block}

# OUTPUT CONTRACT
{target_block}

# PROBLEM-SPECIFIC IMPLEMENTATION HINTS
{hints_block}

Return executable Python code only enclosed in ```python ... ```. Do not write explanations."""
    return [
        {"role": "system", "content": SYMPLANNER_CODEGEN_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def build_symplanner_debug_messages(
    question: str,
    bad_code: str,
    execution_status: str = "error",
    error_tb: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    verification_status: str = "fail",
    verification_feedback: Optional[str] = None,
    planner_note: str = "",
    structured_output: bool = True
) -> List[Dict[str, str]]:
    """Xây dựng thông điệp cho Turn 3: Sửa lỗi mã nguồn có chủ đích (Pure Code Debug)."""
    feedback_lines = []
    if execution_status != "success":
        feedback_lines.append(f"Execution status: {execution_status.upper()}")
        if error_tb:
            clean_tb = str(error_tb).strip()
            if len(clean_tb) > 600:
                clean_tb = clean_tb[-600:]
            feedback_lines.append(f"Traceback:\n{clean_tb}")
    else:
        feedback_lines.append("Execution status: SUCCESS (Code executed without crash)")
        
    if candidate_answer is not None:
        cand_short = str(candidate_answer)[:150]
        feedback_lines.append(f"Candidate answer printed: {cand_short}")
        
    if verification_feedback:
        verif_short = str(verification_feedback)[:600]
        feedback_lines.append(f"Verifier diagnosis: {verif_short}")

    feedback_text = "\n".join(feedback_lines)
    plan_block = planner_note.strip() or "N/A"
    target_block = json.dumps(infer_target_spec(question, planner_note), sort_keys=True)
    hints = build_problem_hints(question)
    hints_block = "\n".join(f"- {hint}" for hint in hints) if hints else "- No extra pattern hint."

    user_text = f"""# PROBLEM
{question}

# PLANNER NOTES
{plan_block}

# OUTPUT CONTRACT
{target_block}

# PROBLEM-SPECIFIC IMPLEMENTATION HINTS
{hints_block}

# PREVIOUS CODE
```python
{str(bad_code).strip()[:1200]}
```

# DIAGNOSIS
{feedback_text}

Fix the issue and return corrected executable Python code only enclosed in ```python ... ```."""

    return [
        {"role": "system", "content": SYMPLANNER_DEBUG_SYSTEM_PROMPT if structured_output else DEBUG_SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ]


def build_prompt_messages(method: str, question: str) -> List[Dict[str, str]]:
    """Xây dựng thông điệp ChatML chuẩn cho Direct, CoT, SymCode và SymPlanner."""
    if method == "SymPlanner":
        return build_planner_messages(question)
    elif method == "SymCode":
        return [
            {"role": "system", "content": SYMCODE_SYSTEM_PROMPT},
            {"role": "user", "content": f"# PROBLEM\n{question}\n# END PROBLEM\n\nReturn executable Python code only enclosed in ```python ... ```."}
        ]
    elif method == "CoT":
        return [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem:\n{question}"}
        ]
    elif method == "Direct":
        return [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem:\n{question}"}
        ]
    else:
        return [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem:\n{question}"}
        ]


# Backward compatibility aliases
def build_retry_prompt_messages(
    question: str,
    prev_code: str,
    execution_status: str = "error",
    error_tb: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    verification_status: str = "fail",
    verification_feedback: Optional[str] = None
) -> List[Dict[str, str]]:
    return build_symplanner_debug_messages(
        question=question,
        bad_code=prev_code,
        execution_status=execution_status,
        error_tb=error_tb,
        candidate_answer=candidate_answer,
        verification_status=verification_status,
        verification_feedback=verification_feedback,
        structured_output=False
    )


def build_symplanner_retry_prompt_messages(
    question: str,
    prev_code: str,
    execution_status: str = "error",
    error_tb: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    verification_status: str = "fail",
    verification_feedback: Optional[str] = None
) -> List[Dict[str, str]]:
    return build_symplanner_debug_messages(
        question=question,
        bad_code=prev_code,
        execution_status=execution_status,
        error_tb=error_tb,
        candidate_answer=candidate_answer,
        verification_status=verification_status,
        verification_feedback=verification_feedback,
        structured_output=True
    )
