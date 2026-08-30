"""Prompt for the Chain-of-Thought baseline."""

from typing import Dict, List

COT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem step-by-step with clear and rigorous logical reasoning.
At the end of your reasoning, write your final answer strictly formatted in \\boxed{answer}."""


def build_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem:\n{question}"},
    ]
