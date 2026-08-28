"""
Prompts for Chain-of-Thought (CoT) reasoning.
"""

from __future__ import annotations

from typing import Dict, List

COT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem step-by-step with clear and rigorous logical reasoning.
At the end of your reasoning, write your final answer strictly formatted in \\boxed{answer}."""


def build_cot_messages(question: str) -> List[Dict[str, str]]:
    """Xay dung prompt ChatML cho phuong phap CoT."""
    return [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem:\n{question}"}
    ]
