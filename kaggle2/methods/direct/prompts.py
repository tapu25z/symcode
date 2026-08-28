"""
Prompts for Direct Zero-shot prediction.
"""

from __future__ import annotations

from typing import Dict, List

DIRECT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the following math problem directly.
Do not provide long explanations. Put only the final answer inside \\boxed{answer}."""


def build_direct_messages(question: str) -> List[Dict[str, str]]:
    """Xay dung prompt ChatML cho phuong phap Direct."""
    return [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem:\n{question}"}
    ]
