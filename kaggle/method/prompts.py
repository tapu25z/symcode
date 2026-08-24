"""
Prompt definitions and message builders for reasoning baselines.
Retains Direct, CoT, and SymCode (Neurosymbolic Equation Solving with SymPy & Verification Loop).
"""

from typing import Dict, List, Optional

# Official SymCode prompt template from SymCode (ACL 2026)
SYMCODE_SYSTEM_PROMPT = """You are an expert mathematical reasoner. Your output must be ONLY a single Python code block fenced as ```python ... ``` with no prose before or after.
Inside that single Python script:
1. Import SymPy with `import sympy as sp`
2. Add explicit step-by-step reasoning as comments throughout your code
3. Document the problem setup:
- Clearly identify variables, constraints, and goals in comments
- Define symbols with appropriate assumptions (e.g., sp.symbols('x', positive=True, integer=True))
4. Include intermediate reasoning steps:
- Each step should have a comment explaining the mathematical reasoning
- Use meaningful variable names that reflect their purpose
- Show the algebraic manipulations clearly
5. For verification:
- Substitute solutions back into original equations
- Check domain constraints (e.g., integer solutions, positive values)
- Filter invalid solutions
6. Print ONLY the final answer in LaTeX boxed form:
print(f"\\\\boxed{{{final_answer}}}")"""

COT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem step-by-step with clear and rigorous logical reasoning.
At the end of your reasoning, write your final answer strictly formatted in \\boxed{answer}."""

DIRECT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem directly.
Do not provide long explanations. Put only the final answer inside \\boxed{answer}."""

SYSTEM_PROMPTS = {
    "Direct": DIRECT_SYSTEM_PROMPT,
    "CoT": COT_SYSTEM_PROMPT,
    "SymCode": SYMCODE_SYSTEM_PROMPT,
}


def build_prompt_messages(method: str, question: str) -> List[Dict[str, str]]:
    """
    Constructs ChatML messages for the specified baseline method.
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
        # Fallback to Direct
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
    Constructs self-debugging ChatML messages with rich feedback for SymCode.
    Feedback includes execution error traceback (if any), candidate answer,
    and mathematical verification diagnosis (without ground truth leakage).
    """
    feedback_lines = []
    
    if execution_status != "success":
        feedback_lines.append(f"### Execution Status: {execution_status.upper()}")
        if error_tb:
            # Keep only the relevant end of traceback (max 800 chars)
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
            "Please carefully review the feedback, fix the logic, constraints, and calculations, "
            "and output ONLY the complete corrected Python script inside a single ```python ... ``` block."
        )}
    ]
