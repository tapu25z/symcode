"""Prompts for the Plan-and-Code (P&C) baseline."""

from typing import Dict, List

PLANCODE_SYSTEM_PROMPT = r"""You are an expert mathematical problem solver and Python code generator.
Solve the problem by first writing a step-by-step plan in Python comments, and then implementing the plan in executable Python code.

Instructions:
1. Return ONLY executable Python code enclosed in a single ```python ... ``` block.
2. Do NOT write explanations outside the code block. Do NOT output <think> tags.
3. At the beginning of the code, write a numbered plan in comments:
   # Plan:
   # Step 1: ...
   # Step 2: ...
4. Directly beneath the plan, write the Python code that executes each planned step.
5. Store the final answer in final_answer and print it formatted strictly as:
   print(f"\\boxed{{{final_answer}}}")
"""


def build_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": PLANCODE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"# PROBLEM\n{question}\n\nFirst write a step-by-step plan in comments, then write the Python code to solve it inside ```python ... ```.",
        },
    ]
