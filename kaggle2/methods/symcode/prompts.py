"""
Prompts for SymCode (Neurosymbolic Equation Solving with SymPy).
"""

from __future__ import annotations

from typing import Dict, List, Optional

SYMCODE_SYSTEM_PROMPT = r"""You are an expert mathematical solver and deterministic Python/SymPy code generator.

Solve the problem by returning ONLY executable Python code enclosed in a single ```python ... ``` block.
Do NOT write explanations. Do NOT output <think> tags.

The code MUST:
1. import sympy as sp (and math, fractions if helpful).
2. Define all given quantities and formulate equations accurately.
3. Solve for the target quantity symbolically or numerically.
4. Never call `.evalf()` on standard Python int/float.
5. Print ONLY the final answer in LaTeX boxed format at the end:
   print(f"\\boxed{{{final_answer}}}")
"""

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
2. Print ONLY the final answer in LaTeX boxed format:
   print(f"\\boxed{{{final_answer}}}")
"""


def build_symcode_messages(question: str) -> List[Dict[str, str]]:
    """Xay dung prompt ChatML cho phuong phap SymCode."""
    return [
        {"role": "system", "content": SYMCODE_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n# END PROBLEM\n\nReturn executable Python code only enclosed in ```python ... ```."}
    ]


def build_symcode_retry_messages(
    question: str,
    prev_code: str,
    execution_status: str = "error",
    error_tb: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    verification_status: str = "fail",
    verification_feedback: Optional[str] = None
) -> List[Dict[str, str]]:
    """Xay dung prompt retry sua code cho SymCode dua tren Traceback va Verifier Feedback."""
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
        feedback_lines.append(f"Candidate answer printed: {str(candidate_answer)[:150]}")

    if verification_feedback:
        feedback_lines.append(f"Verifier diagnosis: {str(verification_feedback)[:600]}")

    diag_text = "\n".join(feedback_lines)

    user_text = f"""# PROBLEM
{question}

# PREVIOUS CODE
```python
{str(prev_code).strip()[:1200]}
```

# DIAGNOSIS
{diag_text}

Fix the issue and return corrected executable Python code only enclosed in ```python ... ```."""

    return [
        {"role": "system", "content": DEBUG_SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ]
