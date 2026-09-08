"""
Định nghĩa hệ thống prompt và bộ tạo thông điệp ChatML cho các phương pháp benchmark:
Direct, CoT, SymCode và SymPlanner (Decoupled Divide-and-Plan Pipeline: Planner -> Pure Codegen -> Debug Repair).
"""

import re
from typing import Dict, List, Optional, Any

from .target_contract import infer_target_spec

# ==============================================================================
# 1. SYMPLANNER PROMPTS (Extract -> Plan -> SymCode)
# ==============================================================================

EXTRACT_SYSTEM_PROMPT = r"""You identify ONLY the Target and Output format of a math problem in under 20 tokens.
Do not copy, rewrite, or summarize the problem. Do not write equations or code.

Return ONLY these two lines:
# Target: <the exact quantity, entity, or expression to find>
# Output: <pick exactly one: number | text | tuple | set | symbolic | base_notation>

Rules for Output type:
- number: any problem asking for a numerical value, ratio, trigonometric value (e.g. tan A, sin x), length, area, angle, or fraction.
- text: names, true/false, or conic classifications (e.g. ellipse, parabola).
- tuple: coordinates (x, y) or ordered pairs.
- set: multiple roots/values.
- symbolic: ONLY when the question explicitly says "in terms of" or "polynomial in".

Example 1:
Problem: If 2x + 5 = 15, find the value of x^2.
# Target: x^2
# Output: number

Example 2:
Problem: In right triangle ABC with angle B = 90, sin A = 2 cos A. What is tan A?
# Target: tan A
# Output: number

Example 3:
Problem: Determine if the graph of (x/2 - 3)^2 + y^2 = 10 is a parabola, circle, ellipse, or hyperbola.
# Target: conic section classification
# Output: text"""

PLANNER_SYSTEM_PROMPT = r"""You write an operational, step-by-step solution plan for a Python/SymPy solver.
You will receive the original problem and the extracted mathematical state.
Return ONLY numbered plan steps.

Rules:
1. Operational blueprint: Name the exact mathematical theorem, formula, or SymPy technique to use (e.g., equate coefficients, compute discriminant, use Vieta's formulas, solve system).
2. Minimalist & direct: Focus strictly on answering the requested target. Do NOT plan exploratory branches, extra case analyses, or degenerate checks unless the problem explicitly asks for them.
3. Do not calculate or reveal final numbers.
4. Do not write Python code.
5. Keep the plan to 2-4 short, concrete steps.

Example:
# PROBLEM
Determine if the graph of (x/2 - 3)^2 + y^2 = 10 is an ellipse, parabola, or hyperbola.
# TARGET & FORMAT
# Target: conic section classification
# Output: text
1. Expand the equation into general conic form Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 and extract coefficients A, B, C.
2. Compute the discriminant B^2 - 4*A*C.
3. If discriminant < 0 and A != C, conclude ellipse; otherwise determine conic type accordingly."""


REPLAN_SYSTEM_PROMPT = r"""You are revising an unsuccessful solution plan for a mathematical problem.
Review the problem, extracted state, previous failed plan, and diagnosis.
Return ONLY revised numbered plan steps.

Rules:
1. Propose an alternative, simpler mathematical formulation that directly avoids the reported failure.
2. Keep the plan short (2-3 steps) and operational.
3. Do not repeat the failed approach.
4. Do not write code or reveal final numeric values."""


# ==============================================================================
# 2. CODEGEN PROMPTS (Turn 3: Sinh mã nguồn Python/SymPy thuần túy 100%)
# ==============================================================================

SYMPLANNER_CODEGEN_SYSTEM_PROMPT = r"""You are an expert mathematical solver and deterministic Python/SymPy code generator.

Return ONLY executable Python code in one ```python ... ``` block. Do not explain.

Rules:
1. 1:1 Plan realization: Implement the plan directly. Each main step in code must start with "# Step <number>:" corresponding to the plan.
2. Direct execution: Do not add extra exploratory checks, duplicate branching, or unprompted edge-case handlers outside the plan.
3. Import sympy as sp. Use exact arithmetic (sp.Rational, sp.Integer); use floats only when explicitly requested.
4. If using sp.solve, handle results safely (safe_solve is available in globals).
5. Use finite loops only. Never use unbounded while loops.
6. When solving for a ratio or trigonometric value (e.g. tan A = sin A / cos A), compute the numerical ratio directly (e.g. sin_val / cos_val). Never print an unevaluated function call like sp.tan(A).
7. At the end, ALWAYS print the final answer enclosed in LaTeX boxed format:
   - For symbolic/mathematical expressions: print(f"\\boxed{{{sp.latex(final_answer)}}}")
   - For text answers: print(f"\\boxed{{{final_answer}}}")"""
SYMCODE_SYSTEM_PROMPT = r"""You are an expert mathematical solver and deterministic Python/SymPy code generator.

Solve the problem by returning ONLY executable Python code enclosed in a single ```python ... ``` block.
Do NOT write explanations. Do NOT output <think> tags.

The code MUST:
1. import sympy as sp (and math, fractions if helpful).
2. Formulate equations accurately and solve directly for the target quantity using SymPy. Keep the code clean, linear, and deterministic.
3. Guard against empty solution lists before indexing (e.g. check if solutions is non-empty).
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

DEBUG_SYSTEM_PROMPT = r"""You are an expert Python/SymPy code repair engineer.
Fix the reported code/verifier issue and return ONLY the corrected executable Python code block.

Rules:
1. Recompute the requested target directly; do not hard-code numbers or repeat crashed code.
2. Use exact arithmetic (sp.Rational, safe_solve) and guard empty solver results before indexing.
3. Ensure finite execution; never use unbounded while loops.
4. Print ONLY the final answer in LaTeX boxed format: print(f"\\boxed{{{sp.latex(final_answer)}}}")"""

SYMPLANNER_DEBUG_SYSTEM_PROMPT = DEBUG_SYSTEM_PROMPT

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
    """Strip <think>...</think> tags. If closing tag is missing (truncated output), salvage content."""
    text = str(text or "")
    if not text:
        return ""
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>", 1)
        after_think = parts[1].strip()
        if after_think:
            return after_think
        inside_think = parts[0].split("<think>", 1)[-1].strip()
        return inside_think
    if "<think>" in text and "</think>" not in text:
        inside = text.split("<think>", 1)[-1].strip()
        return inside
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    return text.strip()


def clean_planner_note(raw_plan: str) -> str:
    """Clean extracted state or plan, removing thinking tags and bounding context length."""
    if not raw_plan or not raw_plan.strip():
        return ""
    text = remove_thinking_tags(raw_plan.strip())
    if not text:
        text = re.sub(r"</?think>", "", raw_plan).strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return text[:1500].strip()


def build_extract_messages(question: str) -> List[Dict[str, str]]:
    """Build Turn 1 messages: identify the Target and Output format only."""
    return [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n\nIdentify the Target and Output format only in two lines."}
    ]


def build_planner_messages(question: str, extraction: str = "") -> List[Dict[str, str]]:
    """Build Turn 2 messages: write an operational plan given the problem and target."""
    extraction_block = extraction.strip() or "No target/format available."
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n\n# TARGET & FORMAT\n{extraction_block}\n\nWrite the operational plan only."}
    ]
def build_replan_messages(
    question: str,
    extraction: str = "",
    prev_plan: str = "",
    failure_feedback: str = ""
) -> List[Dict[str, str]]:
    """Build Level 2 Backtracking messages: request an alternative plan given diagnosis."""
    extraction_block = extraction.strip() or "No extraction available."
    prev_plan_block = prev_plan.strip() or "N/A"
    diag_block = failure_feedback.strip() or "The previous plan led to execution or verification failure."

    user_content = f"""# PROBLEM
{question}

# EXTRACTED STATE
{extraction_block}

# PREVIOUS FAILED PLAN
{prev_plan_block}

# FAILURE DIAGNOSIS
{diag_block}

Formulate a new, alternative numbered solution plan that avoids the above failure mode. Write the revised plan only."""

    return [
        {"role": "system", "content": REPLAN_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
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



def build_symplanner_codegen_messages(question: str, planner_note: str = "", subject: str = "") -> List[Dict[str, str]]:
    """Build Turn 3 messages: generate SymPy code from extraction + plan."""
    plan_block = planner_note.strip() or "No extraction/plan available. Solve directly from the problem."

    target_spec = infer_target_spec(question, planner_note)
    target_block = format_output_requirement(target_spec)
    user_content = f"""# PROBLEM
{question}

# EXTRACTED STATE AND PLAN
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
    structured_output: bool = True,
    subject: str = ""
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

# EXTRACTED STATE AND PLAN
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
        return build_extract_messages(question)
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


