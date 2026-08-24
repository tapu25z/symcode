"""
Prompt definitions and message builders for reasoning baselines.
Faithful to the official SymCode (ACL 2026) and PAL (ICML 2023) / CoT specifications.
"""

from typing import Dict, List, Optional

# Official SymCode prompt template from Listing 1 in SymCode (ACL 2026)
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

# Official PAL prompt: standard Python only (math, fractions, itertools). SymPy is not part of standard PAL.
PAL_SYSTEM_PROMPT = """You are an expert Python programmer and mathematician. Solve the following math problem by writing clean, executable Python code.
Requirements:
1. Wrap your entire code in a single ```python ... ``` block with no other text.
2. Use variables and standard Python libraries (math, fractions, itertools) to compute the exact solution programmatically.
3. At the end of the script, print ONLY the final answer inside \\boxed{...}:
print(f"\\\\boxed{{{result}}}")"""

COT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem step-by-step with clear and rigorous logical reasoning.
At the end of your reasoning, write your final answer strictly formatted in \\boxed{answer}."""

DIRECT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem directly.
Do not provide long explanations. Put only the final answer inside \\boxed{answer}."""

SYSTEM_PROMPTS = {
    "Direct": DIRECT_SYSTEM_PROMPT,
    "CoT": COT_SYSTEM_PROMPT,
    "PAL": PAL_SYSTEM_PROMPT,
    "SymCode": SYMCODE_SYSTEM_PROMPT,
}


def build_prompt_messages(method: str, question: str) -> List[Dict[str, str]]:
    """
    Constructs ChatML messages for the specified baseline method.
    """
    if method in ["SymCode", "SymCode+"]:
        system_content = SYMCODE_SYSTEM_PROMPT
        user_content = f"# PROBLEM\n{question}\n# END PROBLEM"
    elif method == "PAL":
        system_content = PAL_SYSTEM_PROMPT
        user_content = f"Problem:\n{question}"
    elif method == "CoT":
        system_content = COT_SYSTEM_PROMPT
        user_content = f"Problem:\n{question}"
    else:  # Direct
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
    Constructs self-debugging ChatML messages with rich feedback for SymCode+.
    Feedback includes execution error traceback (if any), candidate answer,
    and mathematical verification diagnosis (without ground truth leakage).
    """
    feedback_lines = []
    
    if execution_status != "success":
        feedback_lines.append(f"### Execution Status: {execution_status.upper()}")
        if error_tb:
            feedback_lines.append(f"Traceback:\n```\n{error_tb}\n```")
    else:
        feedback_lines.append("### Execution Status: SUCCESS (Code executed without crash)")
        
    if candidate_answer is not None:
        feedback_lines.append(f"Candidate Answer extracted: `{candidate_answer}`")
        
    if verification_feedback:
        feedback_lines.append(f"Verification Feedback ({verification_status.upper()}):\n{verification_feedback}")

    feedback_text = "\n\n".join(feedback_lines)

    return [
        {"role": "system", "content": SYMCODE_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n# END PROBLEM"},
        {"role": "assistant", "content": f"```python\n{prev_code}\n```"},
        {"role": "user", "content": (
            f"Execution & Verification Diagnosis:\n{feedback_text}\n\n"
            "Please carefully review the feedback, fix the logic, constraints, and calculations, "
            "and output ONLY the complete corrected Python script inside a single ```python ... ``` block."
        )}
    ]
