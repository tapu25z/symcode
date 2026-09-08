"""Prompts for the PaL (Program-aided Language models) baseline (Gao et al., 2023)."""

from typing import Dict, List

PAL_SYSTEM_PROMPT = r"""You are an expert mathematician and computer programmer.
Solve the following math problem by writing Python code that computes the solution.

Instructions:
1. Return ONLY valid, executable Python code inside a single ```python ... ``` block.
2. Do NOT write natural language explanations outside the code block. Do NOT output <think> tags.
3. Use Python modules like math, fractions, or collections if needed.
4. Define the given quantities and carry out the computation step-by-step in code.
5. At the end, print the final answer formatted strictly inside LaTeX \boxed{...}:
   print(f"\\boxed{{{final_answer}}}")
"""


def build_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": PAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"# PROBLEM\n{question}\n\nReturn executable Python code only enclosed in ```python ... ```.",
        },
    ]
