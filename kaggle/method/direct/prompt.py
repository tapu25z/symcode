"""Prompt for the Direct baseline."""

from typing import Dict, List

DIRECT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem directly.
Do not provide long explanations. Put only the final answer inside \\boxed{answer}."""


def build_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem:\n{question}"},
    ]
