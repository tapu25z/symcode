"""
Định nghĩa hệ thống prompt và bộ tạo thông điệp ChatML cho các phương pháp benchmark:
Direct, CoT và SymCode (Neurosymbolic Equation Solving với SymPy và vòng lặp Verifier).
"""

from typing import Dict, List, Optional

# Prompt cho phương pháp SymCode: Lập kế hoạch CoT + Mã nguồn SymPy thực thi
SYMCODE_SYSTEM_PROMPT = """You are an expert mathematical reasoner and symbolic computation specialist.

To solve the problem accurately without missing any variables, constraints, or boundary conditions, follow this two-step process:

### Step 1: Concise Mathematical Breakdown (Chain-of-Thought)
- Briefly explain your step-by-step mathematical reasoning.
- Explicitly identify all given quantities, target unknowns, and define each variable with its domain assumptions (e.g., positive integer, real number).
- Set up the governing algebraic equations and relationships clearly.

### Step 2: Executable SymPy Python Script
- Provide the complete, self-contained Python script enclosed in a single ```python ... ``` block.
- Import SymPy as `import sympy as sp`.
- Define variables using `sp.symbols(...)` with appropriate assumptions (e.g. `positive=True, integer=True`).
- Formulate and solve the equations using `sp.Eq(...)` and `sp.solve(...)`.
- Verify domain constraints and filter out extraneous roots.
- At the end of the script, print ONLY the final answer in LaTeX boxed format:
  print(f"\\\\boxed{{{final_answer}}}")"""

# Prompt cho Chain-of-Thought (CoT)
COT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem step-by-step with clear and rigorous logical reasoning.
At the end of your reasoning, write your final answer strictly formatted in \\boxed{answer}."""

# Prompt cho Zero-shot Direct
DIRECT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem directly.
Do not provide long explanations. Put only the final answer inside \\boxed{answer}."""

SYSTEM_PROMPTS = {
    "Direct": DIRECT_SYSTEM_PROMPT,
    "CoT": COT_SYSTEM_PROMPT,
    "SymCode": SYMCODE_SYSTEM_PROMPT,
}


def build_prompt_messages(method: str, question: str) -> List[Dict[str, str]]:
    """
    Xây dựng danh sách thông điệp ChatML cho phương pháp benchmark tương ứng.
    """
    if method == "SymCode":
        system_content = SYMCODE_SYSTEM_PROMPT
        user_content = f"# PROBLEM\n{question}\n# END PROBLEM"
    elif method == "CoT":
        system_content = COT_SYSTEM_PROMPT
        user_content = f"Problem:\n{question}"
    elif method == "Direct":
        system_content = DIRECT_SYSTEM_PROMPT
        user_content = f"Problem:\n{question}"
    else:
        system_content = DIRECT_SYSTEM_PROMPT
        user_content = f"Problem:\n{question}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


def build_retry_prompt_messages(
    question: str,
    prev_code: str,
    execution_status: str = "error",
    error_tb: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    verification_status: str = "fail",
    verification_feedback: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Xây dựng thông điệp tự sửa lỗi (Self-Debugging) cho SymCode với phản hồi chi tiết:
    Traceback thực thi lỗi, đáp án ứng viên trích xuất được và chẩn đoán từ bộ kiểm chứng toán học độc lập.
    """
    feedback_lines = []
    
    if execution_status != "success":
        feedback_lines.append(f"### Execution Status: {execution_status.upper()}")
        if error_tb:
            clean_tb = str(error_tb).strip()
            if len(clean_tb) > 800:
                clean_tb = clean_tb[-800:]
            feedback_lines.append(f"Traceback:\n```\n{clean_tb}\n```")
    else:
        feedback_lines.append("### Execution Status: SUCCESS (Code executed without crash)")
        
    if candidate_answer is not None:
        cand_short = str(candidate_answer)[:200]
        feedback_lines.append(f"Candidate Answer extracted: `{cand_short}`")
        
    if verification_feedback:
        verif_short = str(verification_feedback)[:800]
        feedback_lines.append(f"Verification Feedback ({verification_status.upper()}):\n{verif_short}")

    feedback_text = "\n\n".join(feedback_lines)

    code_snippet = str(prev_code).strip()
    if len(code_snippet) > 1500:
        code_snippet = code_snippet[:1500]

    return [
        {"role": "system", "content": SYMCODE_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n# END PROBLEM"},
        {"role": "assistant", "content": f"```python\n{code_snippet}\n```"},
        {"role": "user", "content": (
            f"Execution & Verification Diagnosis:\n{feedback_text}\n\n"
            "Please carefully review the diagnosis above. First, briefly explain the root cause and your correction plan. "
            "Then, output the complete corrected Python script enclosed in a single ```python ... ``` block."
        )}
    ]
