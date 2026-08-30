"""Prompts for the SymCode baseline."""

from typing import Any, Dict, List, Optional

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

DEBUG_SYSTEM_PROMPT = r"""You are repairing Python/SymPy code for a math problem.

Return ONLY corrected executable Python code in one ```python ... ``` block.
Fix the reported issue and keep correct code. Do not explain or output <think> tags.

Rules:
- Recompute the target; do not hard-code an answer.
- Use exact arithmetic where possible and handle fragile solver failures.
- Use finite loops only; never use an unbounded while loop.
- Any reasoning comment must start with "# Step <number>:".
- Print only the required final result."""


def build_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYMCODE_SYSTEM_PROMPT},
        {"role": "user", "content": f"# PROBLEM\n{question}\n# END PROBLEM\n\nReturn executable Python code only enclosed in ```python ... ```."},
    ]


def build_retry_messages(
    question: str,
    prev_code: str,
    execution_status: str = "error",
    error_tb: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    verification_status: str = "fail",
    verification_feedback: Optional[str] = None,
) -> List[Dict[str, str]]:
    from ..prompts import build_symplanner_debug_messages

    return build_symplanner_debug_messages(
        question=question,
        bad_code=prev_code,
        execution_status=execution_status,
        error_tb=error_tb,
        candidate_answer=candidate_answer,
        verification_status=verification_status,
        verification_feedback=verification_feedback,
        structured_output=False,
    )
