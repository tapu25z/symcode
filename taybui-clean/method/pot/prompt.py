"""Prompts for the PoT (Program-of-Thought) baseline (Chen et al., 2023)."""

from typing import Dict, List

POT_SYSTEM_PROMPT = r"""You are an expert mathematical problem solver.
Solve the math problem by expressing your step-by-step reasoning inside a Python function named `solve()`.

Instructions:
1. Return ONLY executable Python code enclosed in a single ```python ... ``` block.
2. Do NOT write explanations outside the code block. Do NOT output <think> tags.
3. Structure your code as:
   def solve():
       # Step-by-step variables representing intermediate thoughts
       ...
       return final_answer
   
   ans = solve()
   print(f"\\boxed{{{ans}}}")
4. Each reasoning step should be an explicitly named intermediate variable.
5. Offload arithmetic and complex calculations to Python.
"""


def build_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": POT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"# PROBLEM\n{question}\n\nSolve the problem by writing a structured Python solve() function enclosed in ```python ... ```.",
        },
    ]
