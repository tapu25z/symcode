"""
Định nghĩa hệ thống prompt và bộ tạo thông điệp ChatML cho các phương pháp benchmark:
Direct, CoT, SymCode và SymPlanner (Decoupled Divide-and-Plan Pipeline: Planner -> Pure Codegen -> Debug Repair).
"""

import re
from typing import Dict, List, Optional, Any

from .target_contract import infer_target_spec
from .problem_hints import build_problem_hints

# ==============================================================================
# 1. PLANNER PROMPTS (Turn 1: Phân tích & Lập kế hoạch ngắn gọn, không sinh code)
# ==============================================================================

PLANNER_SYSTEM_PROMPT = r"""You are a mathematical planner for a Python/SymPy solver.

Return ONLY these labeled lines:
# Target: quantity to find
# Given: important values and relations
# Step 1: ...
# Step 2: ...
# Step 3: ...
# Answer type: number|symbolic|tuple|set|matrix|text|base_notation

Rules:
- Keep every line short and factual.
- Do not write Python code, JSON, markdown, or <think> tags.
- Use fewer steps when the problem is simple."""

# ==============================================================================
# 2. CODEGEN PROMPTS (Turn 2: Sinh mã nguồn Python/SymPy thuần túy 100%)
# ==============================================================================

SYMPLANNER_CODEGEN_SYSTEM_PROMPT = r"""You are a Python/SymPy solver.

Return ONLY executable Python code in one ```python ... ``` block. Do not explain.

Rules:
1. Import sympy as sp and json. Use exact arithmetic, especially sp.Rational; use floats only when requested.
2. Solve the requested target, not an intermediate value. Follow the problem and planner steps.
3. If using sp.solve or another fragile solver, handle failure or an empty result. Use a simple bounded fallback only when practical.
4. Use finite loops only. Never use an unbounded while loop.
5. Add a cheap substitution or direct check when it is natural. Do not add a second algorithm just for show.
6. Never print None, Invalid, NaN, undefined variables, debug text, or intermediate values.
7. Any reasoning comment must start with "# Step <number>:".
8. At the end, print exactly one JSON line with keys answer, canonical_answer, answer_type, unit, and variables using json.dumps(..., default=str).
9. Match the OUTPUT REQUIREMENT, including text, symbolic, tuple, set, matrix, and base notation targets."""

SYMCODE_SYSTEM_PROMPT = r"""You are an expert mathematical solver and deterministic Python/SymPy code generator.

Solve the problem by returning ONLY executable Python code enclosed in a single ```python ... ``` block.
Do NOT write explanations. Do NOT output <think> tags.

The code MUST:
1. import sympy as sp (and math, fractions if helpful).
2. Write the solver with two independent paths (Path A: Symbolic/Analytical, Path B: Empirical/Simulation/Search loop) to cross-verify the answer whenever possible.
3. Guard symbolic solving calls (e.g., sp.solve) with try-except blocks. If SymPy fails, automatically fallback to a bounded search loop or numerical optimization.
4. Define all given quantities and formulate equations accurately.
5. Solve for the target quantity symbolically or numerically.
6. Never call `.evalf()` on standard Python int/float.
7. Avoid using sp.solve() or sp.nonlinsolve() on complex nonlinear or multivariate systems of high degree (e.g. degree >= 3 with multiple variables, or equations containing non-rational exponent powers like **(1/3)), as it causes SymPy to hang indefinitely. Use numerical optimization (e.g., scipy.optimize.minimize or fsolve) instead.
8. Never write infinite loops or unbounded while loops (e.g., custom prime generators). Always use finite for loops (e.g., for i in range(10000)) or specify a maximum iteration count to guarantee termination.
9. Print ONLY the final answer in LaTeX boxed format at the end:
   print(f"\\boxed{{{final_answer}}}")
"""

# ==============================================================================
# 3. DEBUG / REPAIR PROMPTS (Turn 3: Sửa lỗi mã nguồn có chủ đích)
# ==============================================================================

DEBUG_SYSTEM_PROMPT = r"""You are repairing Python/SymPy code for a math problem.

Return ONLY corrected executable Python code in one ```python ... ``` block.
Fix the reported issue and keep correct code. Do not explain or output <think> tags.

Rules:
- Recompute the target; do not hard-code an answer.
- Use exact arithmetic where possible and handle fragile solver failures.
- Use finite loops only; never use an unbounded while loop.
- Any reasoning comment must start with "# Step <number>:".
- Print only the required final result."""

SYMPLANNER_DEBUG_SYSTEM_PROMPT = r"""You are repairing Python/SymPy code for a math problem.

Return ONLY corrected executable Python code in one ```python ... ``` block.
Fix the reported execution or verification issue. Recompute the target; do not
hard-code an answer. Keep correct code and make the smallest useful repair.

Rules:
- Use exact arithmetic where possible and handle fragile solver failures.
- Use finite loops only; never use an unbounded while loop.
- Any reasoning comment must start with "# Step <number>:".
- Do not print debug text or intermediate values.
- Print exactly one JSON line with keys answer, canonical_answer, answer_type,
  unit, and variables using json.dumps(..., default=str).
- Match the OUTPUT REQUIREMENT."""

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
        {"role": "user", "content": f"# PROBLEM\n{question}\n\nReturn the labeled plan only."}
    ]


def format_output_requirement(target_spec: Dict[str, Any]) -> str:
    """Render the output contract as short natural-language lines for small models."""
    unit = target_spec.get("unit") if target_spec.get("unit") is not None else "None"
    diagram = "yes" if target_spec.get("diagram_required") else "no"
    return "\n".join([
        "# OUTPUT REQUIREMENT",
        f"- Answer type: {target_spec.get('answer_type', 'number')}",
        f"- Unit: {unit}",
        f"- Diagram relations required: {diagram}",
    ])


def build_symplanner_codegen_messages(question: str, planner_note: str = "") -> List[Dict[str, str]]:
    """Xây dựng thông điệp cho Turn 2: Sinh mã nguồn Python/SymPy từ đề bài + Kế hoạch."""
    plan_block = planner_note.strip() or "Planner note unavailable. Solve directly from first principles."
    target_spec = infer_target_spec(question, planner_note)
    target_block = format_output_requirement(target_spec)
    user_content = f"""# PROBLEM
{question}

# PLANNER NOTES
{plan_block}

# OUTPUT REQUIREMENT
{target_block}

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
    target_block = format_output_requirement(infer_target_spec(question, planner_note))

    user_text = f"""# PROBLEM
{question}

# PLANNER NOTES
{plan_block}

# OUTPUT REQUIREMENT
{target_block}

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
