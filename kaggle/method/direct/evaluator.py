"""Evaluation wrapper for the Direct baseline."""

from typing import Any, Dict, List, Optional


def evaluate(
    dataset: List[Dict[str, Any]],
    llm: Any,
    checkpoint_file: Optional[str] = None,
    save_every: int = 5,
) -> List[Dict[str, Any]]:
    from ..evaluator import evaluate_direct_or_cot

    return evaluate_direct_or_cot("Direct", dataset, llm, checkpoint_file=checkpoint_file, save_every=save_every)
