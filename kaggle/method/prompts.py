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

EXTRACT_SYSTEM_PROMPT = r"""You extract the target and output format for a math solver.

Return ONLY these 2 labeled lines:
# Target: <target variable/quantity/object asked for>
# Output: <number | text | tuple | set | symbolic>

Output type rules:
- number: any numerical answer, ratio, trig value (tan A, sin x), length, area, angle, fraction.
- text: names, true/false, conic classification.
- tuple: coordinates (x, y) or ordered pair.
- set: solution set or set of values.
- symbolic: ONLY when problem explicitly states "in terms of" or "polynomial in".

Rules:
- Do not solve the problem or write code.
- Keep each line short."""

PLANNER_SYSTEM_PROMPT = r"""You write a short SymPy computational plan for a math solver.

Return 2-3 concise bullet points outlining the calculation steps:
1. Define symbols (sp.symbols).
2. Formulate equations (sp.Eq) or direct formulas.
3. Solve (sp.solve) and compute the target value.

Rules:
- Do NOT calculate or reveal the numeric final answer in the plan.
- Do NOT write Python code.
- Keep the plan short and focused on SymPy operations."""

# ==============================================================================
# 2. CODEGEN PROMPTS (Turn 2: Sinh mã nguồn Python/SymPy thuần túy 100%)
# ==============================================================================

SYMPLANNER_CODEGEN_SYSTEM_PROMPT = r"""You are an expert mathematical solver and deterministic Python/SymPy code generator.

Return ONLY executable Python code in one ```python ... ``` block. Do not explain.

Rules:
1. Import sympy as sp. Use exact symbolic arithmetic.
2. IMPORTANT SYMPY & CODE RULES:
   - sp.Rational(p, q) accepts ONLY integer arguments. For expressions or square roots, use division (p / q) or sp.S(p) / q (e.g. 5 / sp.sqrt(80)). Never pass non-integers to sp.Rational.
   - Keep answers in exact symbolic/fractional form by default (e.g. 2, 3/2). Do NOT call .evalf() unless decimal places are explicitly requested.
   - Use sp.together(expr) or sp.cancel(expr) to combine fraction terms into a single fraction before printing.
   - Do NOT filter out negative solutions (sol > 0) unless the problem strictly restricts the domain (e.g. length, count, probability).
   - FOR CONIC CLASSIFICATION: Use poly = sp.Poly(eq, x, y) to get coefficients A = poly.coeff_monomial(x**2), B = poly.coeff_monomial(x*y), C = poly.coeff_monomial(y**2). Calculate disc = B**2 - 4*A*C. If disc < 0: output 'circle' if A == C and B == 0 else 'ellipse'. If disc == 0: output 'parabola'. If disc > 0: output 'hyperbola'.
   - FOR PARAMETERIZED LINES (x, y) = (x0, y0) + t*(vx, vy) -> y = mx + b: Define t, x = sp.symbols('t x'), eliminate t via t_sol = sp.solve(x - (x0 + t*vx), t)[0], substitute into y equation to get y(x), compute m = sp.diff(y(x), x) and b = y(x).subs(x, 0), and print (m, b).
3. Implement the plan as clean, linear Python code.
4. CRITICAL FOR RATIOS / TRIG FUNCTIONS: When computing a ratio or trig function (e.g. tan A = sin A / cos A), solve for the values and compute the ratio directly using arithmetic division (sin_val / cos_val). Never output unevaluated functions like sp.tan(A).
5. Guard fragile sp.solve calls with try-except fallback or bounded numerical/search fallback.
6. Use finite loops only. Never use an unbounded while loop.
7. At the end, print ONLY the final answer in LaTeX boxed format:
   print(f"\\boxed{{{final_answer}}}")"""
SYMCODE_SYSTEM_PROMPT = r"""You are an expert mathematical solver and deterministic Python/SymPy code generator.

Return ONLY executable Python code in a single ```python ... ``` block. Do NOT write explanations or <think> tags.

Rules:
1. Import sympy as sp (and math if needed). Use exact symbolic arithmetic.
2. sp.Rational(p, q) accepts ONLY integer arguments. For expressions or square roots, use division (p / q) or sp.S(p) / q.
3. Keep answers in exact symbolic/fractional form by default. Do NOT call .evalf() unless decimal places are explicitly requested.
4. FOR CONIC CLASSIFICATION: Use poly = sp.Poly(eq, x, y) to get coefficients A = poly.coeff_monomial(x**2), B = poly.coeff_monomial(x*y), C = poly.coeff_monomial(y**2). Calculate disc = B**2 - 4*A*C. If disc < 0: output 'circle' if A == C and B == 0 else 'ellipse'. If disc == 0: output 'parabola'. If disc > 0: output 'hyperbola'.
5. Guard fragile sp.solve calls with simple try-except fallback or bounded numerical fallback.
6. Use finite loops only. Never use an unbounded while loop.
7. At the end, print ONLY the final answer in LaTeX boxed format:
   print(f"\\boxed{{{final_answer}}}")"""

# ==============================================================================
# 3. DEBUG / REPAIR PROMPTS (Turn 3: Sửa lỗi mã nguồn có chủ đích)
# ==============================================================================

DEBUG_SYSTEM_PROMPT = r"""You are repairing Python/SymPy code for a math problem.

Return ONLY corrected executable Python code in one ```python ... ``` block.
Fix the reported issue and keep correct code. Do not explain or output <think> tags.

Rules:
- Recompute the target; do not hard-code an answer.
- sp.Rational(p, q) accepts ONLY integers p and q. For expressions/sqrts, use p / q or sp.S(p) / q.
- Do NOT call .evalf() unless decimal places are requested. Keep exact symbolic expressions.
- FOR CONIC CLASSIFICATION: Use poly = sp.Poly(eq, x, y) to get A, B, C and disc = B**2 - 4*A*C to determine shape.
- FOR PARAMETERIZED LINES: Eliminate parameter t from x equation, substitute into y equation to find y(x) = m*x + b, extract m and b, print (m, b).
- Use sp.together(expr) or sp.cancel(expr) to combine fraction terms.
- Do NOT filter out negative solutions unless the problem domain strictly requires it.
- Use finite loops only; never use an unbounded while loop.
- Any reasoning comment must start with "# Step <number>:".
- Print only the required final result in \boxed{final_answer}."""

SYMPLANNER_DEBUG_SYSTEM_PROMPT = DEBUG_SYSTEM_PROMPT

# ==============================================================================
# 4. BASELINE PROMPTS (Direct & CoT)
# ==============================================================================

COT_SYSTEM_PROMPT = """You are a mathematician. Solve the problem in at most 2-3 brief steps.
Do not write long scratchpads or detailed derivations.
At the end, write your final answer strictly formatted in \\boxed{answer}."""

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


def build_extract_messages(question: str) -> List[Dict[str, str]]:
    """Build Turn 1 messages: extract the mathematical state."""
    return [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n\nExtract target and output type format only."}
    ]


def build_planner_messages(question: str, extraction: str = "") -> List[Dict[str, str]]:
    """Build Turn 2 messages: write a plan from the problem and extraction."""
    extraction_block = extraction.strip() or "No extraction available."
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n\n# EXTRACTED STATE\n{extraction_block}\n\nWrite the plan only."}
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


