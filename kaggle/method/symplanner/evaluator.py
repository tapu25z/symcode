"""Evaluation wrapper for SymPlanner."""

from typing import Any, Dict, List, Optional


def evaluate(
    dataset: List[Dict[str, Any]],
    llm: Any,
    timeout: int = 15,
    max_retries: int = 2,
    checkpoint_file: Optional[str] = None,
    save_every: int = 5,
) -> List[Dict[str, Any]]:
    from ..evaluator import evaluate_symplanner

    return evaluate_symplanner(
        dataset,
        llm,
        timeout=timeout,
        max_retries=max_retries,
        checkpoint_file=checkpoint_file,
        save_every=save_every,
    )
