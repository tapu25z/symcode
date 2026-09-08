"""Evaluation wrapper for PoT (Program-of-Thought)."""

from typing import Any, Dict, List, Optional


def evaluate(
    dataset: List[Dict[str, Any]],
    llm: Any,
    timeout: int = 15,
    max_retries: int = 0,
    checkpoint_file: Optional[str] = None,
    save_every: int = 5,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    from ..evaluator import evaluate_program_aided

    return evaluate_program_aided(
        "PoT",
        dataset,
        llm,
        timeout=timeout,
        max_retries=max_retries,
        checkpoint_file=checkpoint_file,
        save_every=save_every,
        verbose=verbose,
        mode="pot",
    )
