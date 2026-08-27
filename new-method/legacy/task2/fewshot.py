#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tải và chọn few-shot examples cho Module 2/Module 4.

Few-shot bank được index theo problem_type và answer_type để prompt chọn ví dụ
gần bài toán hiện tại hơn, thay vì lấy ngẫu nhiên toàn bộ.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .rag import build_rag_indexes, retrieve_examples


# ===== Extracted notebook cell 12 =====
# Loads pre-generated examples (from data_generator output)
# and selects relevant ones for each LLM call based on
# problem_type (Module 2) or answer_type (Module 4).

def load_few_shot_bank(path: str) -> Dict[str, Any]:
    """
    Load few-shot examples from a JSONL file and build indices.

    Returns a dict with:
      - "all": List of all examples
      - "by_problem_type": {problem_type: [examples]}
      - "by_answer_type":  {answer_type: [examples]}

    Each example must have: question, module2_extract, module3_si_extract,
    module4_symcode, answer_type, detected_problem_type.
    """
    if not os.path.exists(path):
        print(f"WARNING: Few-shot data file not found at {path}", flush=True)
        return {"all": [], "by_problem_type": {}, "by_answer_type": {}}

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # Only include usable examples that have all required fields
                if (rec.get("module2_extract")
                        and rec.get("module4_symcode")
                        and rec.get("usable_for_sft", False)):
                    records.append(rec)
            except Exception:
                pass

    # Build indices for fast lookup
    by_problem_type: Dict[str, List[Dict]] = defaultdict(list)
    by_answer_type:  Dict[str, List[Dict]] = defaultdict(list)

    for rec in records:
        pt = rec.get("detected_problem_type", "unknown")
        at = rec.get("answer_type", "numeric_compute")
        by_problem_type[pt].append(rec)
        by_answer_type[at].append(rec)

    bank = {
        "all": records,
        "by_problem_type": dict(by_problem_type),
        "by_answer_type":  dict(by_answer_type),
        "rag": build_rag_indexes(records),
    }

    print(f"Few-shot bank loaded: {len(records)} usable examples", flush=True)
    print(f"  By problem_type: { {k: len(v) for k, v in by_problem_type.items()} }", flush=True)
    print(f"  By answer_type:  { {k: len(v) for k, v in by_answer_type.items()} }", flush=True)

    return bank


def _normalize_exclude_ids(exclude_ids: Optional[Any]) -> set[str]:
    """Normalize one id or an iterable of ids for leakage-safe retrieval."""
    if exclude_ids is None:
        return set()
    if isinstance(exclude_ids, (str, int)):
        return {str(exclude_ids).strip()}
    return {str(x).strip() for x in exclude_ids if str(x).strip()}


def _filter_excluded_examples(
    examples: List[Dict[str, Any]],
    exclude_ids: Optional[Any],
) -> List[Dict[str, Any]]:
    """Drop examples from the same source id as the current eval record."""
    excluded = _normalize_exclude_ids(exclude_ids)
    if not excluded:
        return list(examples)

    filtered = []
    for ex in examples:
        ex_ids = {
            str(ex.get("id", "")).strip(),
            str(ex.get("source_id", "")).strip(),
        }
        if ex_ids.isdisjoint(excluded):
            filtered.append(ex)
    return filtered


def _dedupe_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for ex in examples:
        key = str(ex.get("id", "")) or str(id(ex))
        if key in seen:
            continue
        seen.add(key)
        unique.append(ex)
    return unique


def _runtime_config() -> Any:
    try:
        from kaggle_pipeline import core as runtime_config
        return runtime_config
    except Exception:
        return None


def _parse_retrieval_query(query: str) -> tuple[str, str]:
    """Extract natural question/target text from Module 4 JSON query when possible."""
    if not query:
        return "", ""
    try:
        payload = json.loads(query)
    except Exception:
        return query, ""
    if not isinstance(payload, dict):
        return query, ""

    question = str(payload.get("question", "") or query)
    target = payload.get("target")
    if isinstance(target, dict):
        bits = [
            str(target.get(key, "")).strip()
            for key in ("name", "symbol", "unit", "answer_type")
            if str(target.get(key, "")).strip()
        ]
        return question, " ".join(bits)
    if target:
        return question, str(target)
    return question, ""


def _retrieve_with_bge(
    query: str,
    n: int,
    problem_type: str = "",
    answer_type: str = "",
    exclude_ids: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    runtime_config = _runtime_config()
    retriever = getattr(runtime_config, "bge_retriever", None) if runtime_config else None
    if not retriever or not query or n <= 0:
        return []

    question, target = _parse_retrieval_query(query)
    hits = retriever.search(
        question=question,
        problem_type=problem_type or None,
        target=target or None,
        top_k=n,
        same_type_first=bool(problem_type),
        exclude_ids=exclude_ids,
    )
    examples = []
    for hit in hits:
        raw = hit.get("raw") or {}
        if answer_type and raw.get("answer_type") and raw.get("answer_type") != answer_type:
            continue
        examples.append(raw)
    return _dedupe_examples(examples)


def format_retrieved_example_for_prompt(example: Dict[str, Any], index: int) -> str:
    """Human-readable few-shot formatting for dense RAG debugging or prompt variants."""
    module2 = example.get("module2_extract") or {}
    plan = example.get("solution_plan") or {}
    execution = example.get("execution") or {}
    code = str(example.get("module4_symcode", "")).strip()
    final_answer = " ".join(
        str(x).strip()
        for x in (execution.get("numeric_answer"), execution.get("pred_unit"))
        if str(x or "").strip()
    )
    return (
        f"[Example {index}]\n"
        f"Problem type: {example.get('detected_problem_type', '')}\n\n"
        f"Question:\n{example.get('question', '')}\n\n"
        f"Extracted given/target:\n{json.dumps(module2, ensure_ascii=False, indent=2)}\n\n"
        f"Solution plan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
        f"Sympy code:\n```python\n{code}\n```\n\n"
        f"Final answer:\n{final_answer}\n"
    ).strip()


def format_retrieved_examples_block(
    examples: List[Dict[str, Any]],
    question: str,
) -> str:
    blocks = [format_retrieved_example_for_prompt(ex, idx) for idx, ex in enumerate(examples, start=1)]
    blocks.append(
        "Now solve the new problem.\n\n"
        f"Question:\n{question}\n\n"
        "Return the required JSON/module output according to the existing pipeline format."
    )
    return "\n\n".join(blocks)


def select_few_shot_for_module2(
    bank: Dict[str, Any],
    problem_type: str,
    n: Optional[int] = None,
    exclude_ids: Optional[Any] = None,
    query: str = "",
) -> List[Dict[str, Any]]:
    """
    Select n few-shot examples for Module 2 (problem extraction).

    Strategy:
      1. Try to find examples with matching problem_type
      2. If not enough, fill from all examples
      3. Random selection for diversity
    """
    if n is None:
        from kaggle_pipeline import core as runtime_config
        n = runtime_config.NUM_FEW_SHOT

    selected: List[Dict[str, Any]] = []
    runtime_config = _runtime_config()
    backend = getattr(runtime_config, "RETRIEVAL_BACKEND", "bm25") if runtime_config else "bm25"
    if backend == "bge" and query:
        selected.extend(_retrieve_with_bge(
            query=query,
            n=n,
            problem_type=problem_type,
            answer_type="numeric_compute",
            exclude_ids=exclude_ids,
        ))
        if len(selected) >= n:
            return selected[:n]

    if query:
        selected.extend(retrieve_examples(
            bank=bank,
            index_name="module2",
            query=query,
            n=n,
            filters={"detected_problem_type": problem_type},
            exclude_ids=exclude_ids,
        ))
        if len(selected) < n:
            selected.extend(retrieve_examples(
                bank=bank,
                index_name="module2",
                query=query,
                n=n - len(selected),
                filters=None,
                exclude_ids=exclude_ids,
            ))
        selected = _dedupe_examples(selected)
        if len(selected) >= n:
            return selected[:n]

    # Try matching problem_type first
    candidates = _filter_excluded_examples(
        bank.get("by_problem_type", {}).get(problem_type, []),
        exclude_ids,
    )

    pool = [ex for ex in candidates if ex not in selected]
    if len(pool) >= n - len(selected):
        selected.extend(random.sample(pool, min(n - len(selected), len(pool))))
        return selected[:n]

    # Not enough matching — supplement from all examples
    all_examples = _filter_excluded_examples(bank.get("all", []), exclude_ids)
    if not all_examples:
        return candidates[:n]

    # Use matching ones first, then fill randomly from the rest
    selected.extend(ex for ex in candidates if ex not in selected)
    remaining = [ex for ex in all_examples if ex not in selected]
    need = n - len(selected)
    if remaining and need > 0:
        selected.extend(random.sample(remaining, min(need, len(remaining))))

    return selected[:n]


def select_few_shot_for_module4(
    bank: Dict[str, Any],
    answer_type: str,
    n: Optional[int] = None,
    exclude_ids: Optional[Any] = None,
    query: str = "",
    problem_type: str = "",
) -> List[Dict[str, Any]]:
    """
    Select n few-shot examples for Module 4 (code generation).

    Strategy:
      1. Match by answer_type (critical — prompt is routed by answer_type)
      2. If not enough, fall back to all examples
    """
    if n is None:
        from kaggle_pipeline import core as runtime_config
        n = runtime_config.NUM_FEW_SHOT

    selected: List[Dict[str, Any]] = []
    runtime_config = _runtime_config()
    backend = getattr(runtime_config, "RETRIEVAL_BACKEND", "bm25") if runtime_config else "bm25"
    if backend == "bge" and query:
        selected.extend(_retrieve_with_bge(
            query=query,
            n=n,
            problem_type=problem_type,
            answer_type=answer_type,
            exclude_ids=exclude_ids,
        ))
        if len(selected) >= n:
            return selected[:n]

    if query:
        selected.extend(retrieve_examples(
            bank=bank,
            index_name="module4",
            query=query,
            n=n,
            filters={
                "detected_problem_type": problem_type,
                "answer_type": answer_type,
            },
            exclude_ids=exclude_ids,
        ))
        if len(selected) < n:
            selected.extend(retrieve_examples(
                bank=bank,
                index_name="module4",
                query=query,
                n=n - len(selected),
                filters={"answer_type": answer_type},
                exclude_ids=exclude_ids,
            ))
        selected = _dedupe_examples(selected)
        if len(selected) >= n:
            return selected[:n]

    candidates = _filter_excluded_examples(
        bank.get("by_answer_type", {}).get(answer_type, []),
        exclude_ids,
    )

    pool = [ex for ex in candidates if ex not in selected]
    if len(pool) >= n - len(selected):
        selected.extend(random.sample(pool, min(n - len(selected), len(pool))))
        return selected[:n]

    # Fallback: use whatever is available
    all_examples = _filter_excluded_examples(bank.get("all", []), exclude_ids)
    if not all_examples:
        return candidates[:n]

    selected.extend(ex for ex in candidates if ex not in selected)
    remaining = [ex for ex in all_examples if ex not in selected]
    need = n - len(selected)
    if remaining and need > 0:
        selected.extend(random.sample(remaining, min(need, len(remaining))))

    return selected[:n]
